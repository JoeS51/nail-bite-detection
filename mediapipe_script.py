import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cam = cv.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()

    if not success:
        print("Camera Frame not available")
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    hands_detected = hands.process(frame)
    face_detected = face_mesh.process(frame)

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

    if face_detected.multi_face_landmarks:
        for face_landmarks in face_detected.multi_face_landmarks:
            drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing.DrawingSpec(color=(0,255,0), thickness=2)
            )

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()