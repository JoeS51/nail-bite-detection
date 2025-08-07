import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for processing video frames
    max_num_hands=2,           # Maximum number of hands to detect
    min_detection_confidence=0.5  # Minimum confidence threshold for hand detection
)

cam = cv.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()

    if not success:
        print("Camera Frame not available")
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    hands_detected = hands.process(frame)

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()