# Nail Bite Detection

I have a bad habit of biting my nails. The only solution I could think of was to shock myself whenever I try to bite them, so I built a computer vision system that can tell when my hand moves toward my mouth.

The project uses a webcam, MediaPipe, and OpenCV to track my fingertips and lips in real time. If a fingertip gets close enough to my mouth, the program displays a warning and sends an HTTP request to a connected device. The current prototype turns on an LED for three seconds, which provides a safe output for testing the detection system.

> **Safety:** The code in this repository controls an LED only. Do not connect homemade electrical devices to your body.

## Run the project

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Update `BASE_URL` in `mediapipe_script.py` to match your device's address.

3. Start the detector:

   ```bash
   python mediapipe_script.py
   ```

Press `q` to stop the program.
