import subprocess

command = [
    "speaker-test",
    "-D", "plughw:3,0",
    "-c", "2",
    "-t", "sine",
    "-f", "1000",
    "-l", "1"
]

subprocess.run(command, check=True)
