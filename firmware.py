import asyncio
import json
import os
import signal
import subprocess
import time
from typing import Optional

import board
import httpx
import neopixel
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


ML_WEBRTC_OFFER_URL = "https://wildsafe-ml-service.onrender.com/predict/webrtc/offer"
ORCHESTRATOR_EVENTS_URL = "https://smart-wild.onrender.com/events"

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
        codec for codec in capabilities.codecs if codec.mimeType.lower() == "video/h264"
    ]
    if h264_codecs:
        transceiver.setCodecPreferences(h264_codecs)


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


async def start_webrtc_stream():
    global peer_connection, rpicam_process

    if peer_connection is not None:
        return

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

    pc = RTCPeerConnection()
    peer_connection = pc
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

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(ML_WEBRTC_OFFER_URL, json=payload)
        response.raise_for_status()
        answer = response.json()

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer.get("type", "answer"))
    )
    print(f"Streaming WebRTC video. stream_id={answer.get('stream_id')}")


async def start_webrtc_stream_with_log():
    try:
        await start_webrtc_stream()
    except Exception as exc:
        print(f"WebRTC stream failed: {exc}")


def incident_frequency(incident: dict) -> Optional[int]:
    location = incident.get("location") or {}
    if location.get("camera_id") != CAMERA_ID:
        return None

    frequency = incident.get("speaker_frequency_hz") or incident.get("frequency_hz")
    if frequency:
        return int(frequency)

    return INCIDENT_FREQUENCIES.get(incident.get("type"))


async def listen_for_alerts():
    while True:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", ORCHESTRATOR_EVENTS_URL) as response:
                    response.raise_for_status()
                    event_name = "message"
                    data_lines = []

                    async for line in response.aiter_lines():
                        if line == "":
                            if event_name == "incident" and data_lines:
                                incident = json.loads("\n".join(data_lines))
                                frequency = incident_frequency(incident)
                                if frequency:
                                    await asyncio.to_thread(alert_hardware, frequency)
                            event_name = "message"
                            data_lines = []
                            continue

                        if line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            event_name = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:") :].strip())
        except Exception as exc:
            print(f"SSE alert stream disconnected: {exc}")
            await asyncio.sleep(5)


async def shutdown():
    led_off()
    if peer_connection is not None:
        await peer_connection.close()
    if rpicam_process is not None and rpicam_process.poll() is None:
        rpicam_process.terminate()


async def main():
    led_off()
    video_task = asyncio.create_task(start_webrtc_stream_with_log())
    alerts_task = asyncio.create_task(listen_for_alerts())
    try:
        await asyncio.gather(video_task, alerts_task)
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
