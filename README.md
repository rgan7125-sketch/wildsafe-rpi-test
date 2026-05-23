# Smart Wild
Smart wild is a survellience service, which detects anomalies, especially those concerning road-side safeties, and alerts the users of Google Maps and authorities. This is a submission to IEEE Quarterly Projects.

This repository is the firmware code for raspberry pi. The responsibilities of the RPi4B include:

1. Continuously stream camera feed as H.264 encodings over WebRTC to https://wildsafe-ml-service.onrender.com/
2. Listen for HTTP requests inbound which will contain a frequency.
3. On HTTP request reception, play sound at the given frequency on the speaker and flash LED lights, both for 5 seconds.
