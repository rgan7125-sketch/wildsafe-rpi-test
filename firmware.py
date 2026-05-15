import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import board
import httpx
import neopixel
import websockets
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaPlayer
from aiortc.rtcicetransport import parse_stun_turn_uri


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [rpi] %(message)s",
)
logger = logging.getLogger("wildsafe.rpi")


def load_env_file(path: str = ".env") -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False

    loaded = False

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value
        loaded = True

    return loaded


ENV_FILE_LOADED = load_env_file()

ML_WEBRTC_OFFER_URL = "https://wildsafe-ml-service-4z6c.onrender.com/predict/webrtc/offer"
ML_HEALTH_URL = "https://wildsafe-ml-service-4z6c.onrender.com/health"
ORCHESTRATOR_EVENTS_URL = "https://smart-wild.onrender.com/events"
ORCHESTRATOR_WS_URL = "wss://smart-wild.onrender.com/handshake"

PIXEL_PIN = board.D18
NUM_PIXELS = 9
BRIGHTNESS = 0.15
ORDER = neopixel.GRB
AUDIO_DEVICE = "plughw:3,0"
ALERT_SECONDS = 5
WEBRTC_SAMPLE_FPS = 3.0
WEBRTC_CONFIDENCE_THRESHOLD = 0.1
WEBRTC_USE_POSE_DETECTION = False

CAMERA_ID = "rpi-roadside-001"
LATITUDE = 37.7749
LONGITUDE = -122.4194
ROAD_NAME = "CA-1"
DIRECTION = "northbound"
MILE_MARKER = "12.4"

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=False,
    pixel_order=ORDER,
)

peer_connection = None
rpicam_process = None


INCIDENT_FREQUENCIES = {
    "animal_on_road": 20000,
    "person_on_road": 1000,
    "stopped_vehicle": 1000,
    "road_obstruction": 1000,
    "unknown": 1000,
}


def led_off():
    pixels.fill((0, 0, 0))
    pixels.show()


def led_alert(on: bool):
    pixels.fill((255, 80, 0) if on else (0, 0, 0))
    pixels.show()


def alert_hardware(freq_hz: int):
    command = [
        "speaker-test",
        "-D",
        AUDIO_DEVICE,
        "-c",
        "2",
        "-t",
        "sine",
        "-f",
        str(freq_hz),
    ]

    tone = subprocess.Popen(command, start_new_session=True)
    try:
        end_at = time.monotonic() + ALERT_SECONDS
        on = True
        while time.monotonic() < end_at:
            led_alert(on)
            on = not on
            time.sleep(0.2)
    finally:
        led_off()
        if tone.poll() is None:
            os.killpg(tone.pid, signal.SIGTERM)
            tone.wait()


def prefer_h264(transceiver):
    capabilities = RTCRtpSender.getCapabilities("video")
    h264_codecs = [
        codec for codec in capabilities.codecs
        if codec.mimeType.lower() == "video/h264"
    ]
    if h264_codecs:
        transceiver.setCodecPreferences(h264_codecs)


def summarize_ice_url(url: str) -> str:
    parsed = urlparse(url)
    transport = parse_qs(parsed.query).get("transport", ["default"])[0]
    host_port = parsed.netloc or parsed.path
    return f"{parsed.scheme}:{host_port} transport={transport}"


def summarize_ice_servers(servers: list[RTCIceServer]) -> list[dict]:
    summaries = []
    for server in servers:
        urls = server.urls if isinstance(server.urls, list) else [server.urls]
        summaries.append(
            {
                "urls": [summarize_ice_url(url) for url in urls],
                "username": "set" if server.username else "unset",
                "credential": "set" if server.credential else "unset",
            }
        )
    return summaries


def summarize_sdp(sdp: str) -> dict:
    lines = sdp.splitlines()
    return {
        "bytes": len(sdp),
        "lines": len(lines),
        "candidates": sum(1 for line in lines if line.startswith("a=candidate:")),
        "media": [line for line in lines if line.startswith("m=")],
    }


def normalize_ice_urls(server: dict) -> list[str]:
    raw_urls = server.get("urls", server.get("url"))
    if raw_urls is None:
        raise ValueError("ICE server entry must include 'urls' or 'url'")

    urls = raw_urls if isinstance(raw_urls, list) else [raw_urls]
    normalized_urls = []
    for url in urls:
        if not isinstance(url, str):
            raise ValueError("ICE server urls must be strings")

        if url.startswith("stun:") and "?transport=" in url:
            url = url.split("?transport=", 1)[0]

        parse_stun_turn_uri(url)
        normalized_urls.append(url)

    return normalized_urls


def load_ice_servers() -> list[RTCIceServer]:
    raw_config = os.getenv("WEBRTC_ICE_SERVERS")
    if raw_config:
        servers = json.loads(raw_config)
    else:
        servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    if not isinstance(servers, list):
        raise ValueError("WEBRTC_ICE_SERVERS must be a JSON array")

    return [
        RTCIceServer(
            urls=normalize_ice_urls(server),
            username=server.get("username"),
            credential=server.get("credential"),
        )
        for server in servers
    ]


async def wait_for_ice_gathering(pc: RTCPeerConnection):
    if pc.iceGatheringState == "complete":
        logger.info("ICE gathering already complete")
        return

    complete = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        logger.info("ICE gathering state changed state=%s", pc.iceGatheringState)
        if pc.iceGatheringState == "complete":
            complete.set()

    try:
        await asyncio.wait_for(complete.wait(), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("ICE gathering timed out; sending current WebRTC offer")


async def wait_for_ml_service():
    logger.info("Warming ML service before WebRTC offer url=%s", ML_HEALTH_URL)
    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, 13):
            try:
                start = time.perf_counter()
                response = await client.get(ML_HEALTH_URL)
                response.raise_for_status()
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "ML service is reachable status=%s elapsed_ms=%.1f",
                    response.status_code,
                    elapsed_ms,
                )
                return
            except Exception as exc:
                delay = min(5 * attempt, 30)
                logger.warning(
                    "ML service not ready attempt=%s/12 retry_delay_s=%s error=%r",
                    attempt,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    raise RuntimeError("ML service did not become reachable")


async def wait_for_webrtc_disconnect(pc: RTCPeerConnection):
    while pc.connectionState not in {"failed", "disconnected", "closed"}:
        await asyncio.sleep(5)

    logger.error("WebRTC connection ended state=%s", pc.connectionState)
    raise RuntimeError(f"WebRTC connection ended: {pc.connectionState}")


def log_startup_config(ice_servers: list[RTCIceServer]):
    logger.info(
        "RPI WebRTC config env_file_loaded=%s webrtc_ice_servers_present=%s "
        "ice_server_count=%s camera_id=%s ml_offer_url=%s sample_fps=%.2f "
        "confidence_threshold=%.2f use_pose_detection=%s",
        ENV_FILE_LOADED,
        bool(os.getenv("WEBRTC_ICE_SERVERS")),
        len(ice_servers),
        CAMERA_ID,
        ML_WEBRTC_OFFER_URL,
        WEBRTC_SAMPLE_FPS,
        WEBRTC_CONFIDENCE_THRESHOLD,
        WEBRTC_USE_POSE_DETECTION,
    )
    logger.info("RPI ICE servers summary=%s", summarize_ice_servers(ice_servers))


async def start_webrtc_stream():
    global peer_connection, rpicam_process

    if peer_connection is not None:
        logger.info("WebRTC startup skipped because peer connection already exists")
        return

    await wait_for_ml_service()

    logger.info("Starting rpicam-vid process codec=h264 inline=true output=stdout")
    rpicam_process = subprocess.Popen(
        [
            "rpicam-vid",
            "-t",
            "0",
            "--codec",
            "h264",
            "--inline",
            "-o",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    logger.info("rpicam-vid started pid=%s", rpicam_process.pid)

    player = MediaPlayer(rpicam_process.stdout, format="h264")
    if player.video is None:
        raise RuntimeError("rpicam-vid did not produce a video track")
    logger.info("MediaPlayer video track ready kind=%s", player.video.kind)

    ice_servers = load_ice_servers()
    log_startup_config(ice_servers)

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(iceServers=ice_servers)
        if ice_servers
        else None
    )
    peer_connection = pc

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("WebRTC connection state changed state=%s", pc.connectionState)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state changed state=%s", pc.iceConnectionState)

    @pc.on("signalingstatechange")
    async def on_signalingstatechange():
        logger.info("WebRTC signaling state changed state=%s", pc.signalingState)

    transceiver = pc.addTransceiver(player.video, direction="sendonly")
    prefer_h264(transceiver)
    logger.info("Video transceiver added direction=sendonly codec_preference=h264")

    offer = await pc.createOffer()
    logger.info("WebRTC SDP offer created summary=%s", summarize_sdp(offer.sdp))
    await pc.setLocalDescription(offer)
    logger.info(
        "Local description set type=%s summary=%s",
        pc.localDescription.type,
        summarize_sdp(pc.localDescription.sdp),
    )
    await wait_for_ice_gathering(pc)

    payload = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sample_fps": WEBRTC_SAMPLE_FPS,
        "confidence_threshold": WEBRTC_CONFIDENCE_THRESHOLD,
        "camera_id": CAMERA_ID,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "road_name": ROAD_NAME,
        "direction": DIRECTION,
        "mile_marker": MILE_MARKER,
        "use_pose_detection": WEBRTC_USE_POSE_DETECTION,
    }

    timeout = httpx.Timeout(connect=20, read=180, write=20, pool=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        logger.info(
            "Posting WebRTC offer url=%s payload_summary=%s",
            ML_WEBRTC_OFFER_URL,
            {
                "type": payload["type"],
                "camera_id": payload["camera_id"],
                "sample_fps": payload["sample_fps"],
                "confidence_threshold": payload["confidence_threshold"],
                "use_pose_detection": payload["use_pose_detection"],
                "sdp": summarize_sdp(payload["sdp"]),
            },
        )
        start = time.perf_counter()
        try:
            response = await client.post(ML_WEBRTC_OFFER_URL, json=payload)
            response.raise_for_status()
        except Exception:
            logger.exception("WebRTC offer POST failed url=%s", ML_WEBRTC_OFFER_URL)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "WebRTC offer POST succeeded status=%s elapsed_ms=%.1f",
            response.status_code,
            elapsed_ms,
        )
        answer = response.json()

    stream_id = answer.get("stream_id")
    logger.info(
        "WebRTC answer received stream_id=%s type=%s summary=%s",
        stream_id,
        answer.get("type", "answer"),
        summarize_sdp(answer["sdp"]),
    )
    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=answer["sdp"],
            type=answer.get("type", "answer"),
        )
    )
    logger.info("Remote description set stream_id=%s", stream_id)
    logger.info("Streaming WebRTC video stream_id=%s", stream_id)
    await wait_for_webrtc_disconnect(pc)


async def start_webrtc_stream_with_log():
    retry_delay = 5
    while True:
        try:
            await cleanup()
            await start_webrtc_stream()
        except Exception as exc:
            logger.exception("WebRTC stream failed error=%r", exc)
            logger.info("Retrying WebRTC startup retry_delay_s=%s", retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)


def incident_frequency(incident: dict) -> Optional[int]:
    location = incident.get("location") or {}
    if location.get("camera_id") != CAMERA_ID:
        return None

    frequency = incident.get("speaker_frequency_hz") or incident.get("frequency_hz")
    if frequency:
        return int(frequency)

    return INCIDENT_FREQUENCIES.get(incident.get("type"))


async def process_incident(raw_data: str):
    logger.info("Received incident event raw_data=%s", raw_data)

    try:
        incident = json.loads(raw_data)
        frequency = incident_frequency(incident)
        if frequency:
            logger.info("Triggering hardware alert frequency_hz=%s", frequency)
            await asyncio.to_thread(alert_hardware, frequency)
        else:
            logger.info(
                "Incident received but ignored reason=camera_mismatch_or_no_frequency"
            )
    except Exception as inner_exc:
        logger.exception("Failed to process incident payload error=%r", inner_exc)


async def listen_for_alerts_sse():
    timeout = httpx.Timeout(connect=10, read=35, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", ORCHESTRATOR_EVENTS_URL) as response:
            response.raise_for_status()
            logger.info("Connected to SSE alert stream url=%s", ORCHESTRATOR_EVENTS_URL)

            event_name = "message"
            data_lines = []

            async for line in response.aiter_lines():
                if line == "":
                    if event_name == "incident" and data_lines:
                        await process_incident("\n".join(data_lines))

                    event_name = "message"
                    data_lines = []
                    continue

                if line.startswith(":"):
                    continue

                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())


async def listen_for_alerts_websocket():
    async with websockets.connect(
        ORCHESTRATOR_WS_URL,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        logger.info("Connected to websocket alert stream url=%s", ORCHESTRATOR_WS_URL)
        async for message in websocket:
            await process_incident(message)


async def listen_for_alerts():
    while True:
        try:
            await listen_for_alerts_sse()
        except Exception as exc:
            logger.exception("SSE alert stream disconnected error=%r", exc)

        try:
            await listen_for_alerts_websocket()
        except Exception as exc:
            logger.exception("Websocket alert stream disconnected error=%r", exc)

        logger.info("Alert stream reconnecting retry_delay_s=2")
        await asyncio.sleep(2)


async def cleanup():
    global peer_connection, rpicam_process

    logger.info("Cleanup started")
    try:
        led_off()
    except Exception:
        logger.exception("Failed to turn LED off during cleanup")

    if peer_connection is not None:
        logger.info("Closing peer connection state=%s", peer_connection.connectionState)
        await peer_connection.close()
        peer_connection = None
        logger.info("Peer connection closed")

    if rpicam_process is not None:
        logger.info("Stopping rpicam process pid=%s", rpicam_process.pid)
        if rpicam_process.poll() is None:
            rpicam_process.terminate()
            try:
                rpicam_process.wait(timeout=5)
                logger.info("rpicam process terminated pid=%s", rpicam_process.pid)
            except subprocess.TimeoutExpired:
                logger.warning("rpicam process did not terminate; killing pid=%s", rpicam_process.pid)
                rpicam_process.kill()
                logger.info("rpicam process killed pid=%s", rpicam_process.pid)
        rpicam_process = None
    logger.info("Cleanup finished")


async def main():
    led_off()

    webrtc_task = asyncio.create_task(start_webrtc_stream_with_log())
    alerts_task = asyncio.create_task(listen_for_alerts())

    try:
        await asyncio.gather(webrtc_task, alerts_task)
    finally:
        await cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down firmware")
