import time
import subprocess

import board
import neopixel

# --------------------
# Mock API response
# --------------------
test_data = {
    "alert": True,
    "speaker_frequency_hz":20000
}

# --------------------
# LED settings
# --------------------
PIXEL_PIN = board.D18
NUM_PIXELS = 8
BRIGHTNESS = 0.15
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=False,
    pixel_order=ORDER
)

AUDIO_DEVICE = "plughw:3,0"


def led_off():
    pixels.fill((0, 0, 0))
    pixels.show()


def led_alert():
    pixels.fill((255, 80, 0))
    pixels.show()


def play_tone(freq_hz: int):
    command = [
        "speaker-test",
        "-D", AUDIO_DEVICE,
        "-c", "2",
        "-t", "sine",
        "-f", str(freq_hz),
        "-l", "1"
    ]
    subprocess.run(command, check=True)


def main():
    try:
        led_off()

        if test_data["alert"]:
            freq = int(test_data["speaker_frequency_hz"])
            print(f"ALERT received. Playing tone at {freq} Hz")
            led_alert()
            play_tone(freq)
            time.sleep(1)
        else:
            print("No alert. No action needed.")
            led_off()

    finally:
        led_off()


if __name__ == "__main__":
    main()
