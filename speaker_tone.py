import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python3 speaker_tone.py <frequency>")
    sys.exit(1)

frequency = sys.argv[1]

command = [
    "speaker-test",
    "-D", "plughw:3,0",
    "-c", "2",
    "-t", "sine",
    "-f", frequency,
    "-l", "1"
]

subprocess.run(command, check=True)
