import asyncio
import os
import signal
import subprocess
import time
from typing import Optional

import board
import httpx
import neopixel
import uvicorn
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ML_WEBRTC_OFFER_URL = "https://wildsafe-ml-service.onrender.com/predict/webrtc/offer"

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

app = FastAPI(title="Smart Wild RPi Firmware")
peer_connection = None
rpicam_process = None


class AlertRequest(BaseModel):
    frequency: Optional[int] = None
    frequency_hz: Optional[int] = None
    speaker_frequency_hz: Optional[int] = None

    def get_frequency(self) -> int:
        freq = self.frequency_hz or self.speaker_frequency_hz or self.frequency
        if not freq:
            raise HTTPException(
                status_code=422,
                detail="Expected frequency_hz, speaker_frequency_hz, or frequency",
            )
        return int(freq)


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


@app.on_event("startup")
async def startup():
    led_off()
    asyncio.create_task(start_webrtc_stream_with_log())


@app.on_event("shutdown")
async def shutdown():
    led_off()
    if peer_connection is not None:
        await peer_connection.close()
    if rpicam_process is not None and rpicam_process.poll() is None:
        rpicam_process.terminate()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/alert")
async def alert(request: AlertRequest):
    freq = request.get_frequency()
    await asyncio.to_thread(alert_hardware, freq)
    return {"status": "ok", "frequency_hz": freq}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
