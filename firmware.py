import asyncio
import json
import os
import signal
import subprocess
import time
import traceback
from typing import Optional

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


ML_WEBRTC_OFFER_URL = "https://wildsafe-ml-service.onrender.com/predict/webrtc/offer"
ML_HEALTH_URL = "https://wildsafe-ml-service.onrender.com/health"
ORCHESTRATOR_EVENTS_URL = "https://smart-wild.onrender.com/events"
ORCHESTRATOR_WS_URL = "wss://smart-wild.onrender.com/handshake"

PIXEL_PIN = board.D18
NUM_PIXELS = 9
BRIGHTNESS = 0.15
ORDER = neopixel.GRB
AUDIO_DEVICE = "plughw:3,0"
ALERT_SECONDS = 5

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


def load_ice_servers() -> list[RTCIceServer]:
    raw_config = os.getenv("WEBRTC_ICE_SERVERS")
    if raw_config:
        servers = json.loads(raw_config)
    else:
        servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    return [
        RTCIceServer(
            urls=server["urls"],
            username=server.get("username"),
            credential=server.get("credential"),
        )
        for server in servers
    ]


async def wait_for_ice_gathering(pc: RTCPeerConnection):
    if pc.iceGatheringState == "complete":
        return

    complete = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        if pc.iceGatheringState == "complete":
            complete.set()

    try:
        await asyncio.wait_for(complete.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("ICE gathering timed out; sending current WebRTC offer")


async def wait_for_ml_service():
    print("Warming ML service before WebRTC offer")
    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, 13):
            try:
                response = await client.get(ML_HEALTH_URL)
                response.raise_for_status()
                print("ML service is reachable")
                return
            except Exception as exc:
                delay = min(5 * attempt, 30)
                print(
                    f"ML service not ready yet "
                    f"(attempt {attempt}/12): {repr(exc)}; retrying in {delay}s"
                )
                await asyncio.sleep(delay)

    raise RuntimeError("ML service did not become reachable")


async def wait_for_webrtc_disconnect(pc: RTCPeerConnection):
    while pc.connectionState not in {"failed", "disconnected", "closed"}:
        await asyncio.sleep(5)

    raise RuntimeError(f"WebRTC connection ended: {pc.connectionState}")


async def start_webrtc_stream():
    global peer_connection, rpicam_process

    if peer_connection is not None:
        return

    await wait_for_ml_service()

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

    player = MediaPlayer(rpicam_process.stdout, format="h264")
    if player.video is None:
        raise RuntimeError("rpicam-vid did not produce a video track")

    ice_servers = load_ice_servers()
    print(f"Using {len(ice_servers)} ICE server(s)")

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(iceServers=ice_servers)
        if ice_servers
        else None
    )
    peer_connection = pc

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC connection state: {pc.connectionState}")

    transceiver = pc.addTransceiver(player.video, direction="sendonly")
    prefer_h264(transceiver)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await wait_for_ice_gathering(pc)

    payload = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sample_fps": 3.0,
        "confidence_threshold": 0.1,
        "camera_id": CAMERA_ID,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "road_name": ROAD_NAME,
        "direction": DIRECTION,
        "mile_marker": MILE_MARKER,
        "use_pose_detection": False,
    }

    timeout = httpx.Timeout(connect=20, read=180, write=20, pool=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(ML_WEBRTC_OFFER_URL, json=payload)
        response.raise_for_status()
        answer = response.json()

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=answer["sdp"],
            type=answer.get("type", "answer"),
        )
    )
    print(f"Streaming WebRTC video. stream_id={answer.get('stream_id')}")
    await wait_for_webrtc_disconnect(pc)


async def start_webrtc_stream_with_log():
    retry_delay = 5
    while True:
        try:
            await cleanup()
            await start_webrtc_stream()
        except Exception as exc:
            print(f"WebRTC stream failed: {repr(exc)}")
            traceback.print_exc()
            print(f"Retrying WebRTC startup in {retry_delay}s")
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
    print(f"Received incident event: {raw_data}")

    try:
        incident = json.loads(raw_data)
        frequency = incident_frequency(incident)
        if frequency:
            print(f"Triggering hardware alert at {frequency} Hz")
            await asyncio.to_thread(alert_hardware, frequency)
        else:
            print(
                "Incident received, but camera_id did not match "
                "or no frequency was available"
            )
    except Exception as inner_exc:
        print(f"Failed to process incident payload: {repr(inner_exc)}")
        traceback.print_exc()


async def listen_for_alerts_sse():
    timeout = httpx.Timeout(connect=10, read=35, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", ORCHESTRATOR_EVENTS_URL) as response:
            response.raise_for_status()
            print("Connected to SSE alert stream")

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
        print("Connected to websocket alert stream")
        async for message in websocket:
            await process_incident(message)


async def listen_for_alerts():
    while True:
        try:
            await listen_for_alerts_sse()
        except Exception as exc:
            print(f"SSE alert stream disconnected: {repr(exc)}")
            traceback.print_exc()

        try:
            await listen_for_alerts_websocket()
        except Exception as exc:
            print(f"Websocket alert stream disconnected: {repr(exc)}")
            traceback.print_exc()

        print("Alert stream reconnecting in 2 seconds")
        await asyncio.sleep(2)


async def cleanup():
    global peer_connection, rpicam_process

    try:
        led_off()
    except Exception:
        pass

    if peer_connection is not None:
        await peer_connection.close()
        peer_connection = None

    if rpicam_process is not None:
        if rpicam_process.poll() is None:
            rpicam_process.terminate()
            try:
                rpicam_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rpicam_process.kill()
        rpicam_process = None


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
        print("Shutting down firmware")
