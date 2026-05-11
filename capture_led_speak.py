import os
import time
import subprocess
from datetime import datetime

import board
import neopixel

PIXEL_PIN = board.D18
NUM_PIXELS = 9
BRIGHTNESS = 0.15
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=False,
    pixel_order=ORDER
)

SAVE_DIR = "/home/dt/wildlife_node/captures"
os.makedirs(SAVE_DIR, exist_ok=True)

AUDIO_DEVICE = "plughw:3,0"


def led_off():
    pixels.fill((0, 0, 0))
    pixels.show()


def led_color(color):
    pixels.fill(color)
    pixels.show()


def capture_image() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")

    cmd = [
        "rpicam-still",
        "-n",
        "-o", filename
    ]
    subprocess.run(cmd, check=True)
    return filename


def speak_usb(text: str):
    # espeak-ng outputs WAV data to stdout, then aplay sends it to the USB speaker
    espeak = subprocess.Popen(
        ["espeak-ng", "--stdout", text],
        stdout=subprocess.PIPE
    )

    aplay = subprocess.Popen(
        ["aplay", "-D", AUDIO_DEVICE],
        stdin=espeak.stdout
    )

    if espeak.stdout is not None:
        espeak.stdout.close()

    aplay.wait()
    espeak.wait()


def main():
    try:
        led_off()

        # Blue while capturing
        led_color((0, 0, 255))
        image_path = capture_image()

        # Green while speaking
        led_color((0, 255, 0))
        speak_usb("Capture complete")

        print(f"Saved image: {image_path}")
        time.sleep(1)

    finally:
        led_off()


if __name__ == "__main__":
    main()
