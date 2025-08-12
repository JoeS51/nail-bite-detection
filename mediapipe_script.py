import cv2 as cv
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

import requests
import time
BASE_URL = "http://192.168.1.36"

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

# 4 = thumb, 8 = index, 12 = middle, 16 = ring, 20 = pinky
fingertip_indices = [4, 8, 12, 16, 20]

lip_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

cam = cv.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hands_detected = hands.process(frame_rgb)
    face_detected = face_mesh.process(frame_rgb)
    frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    h, w, _ = frame.shape
    fingertip_points = []
    lip_polygon_pts = None

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            for finger in fingertip_indices:
                lm = hand_landmarks.landmark[finger]
                x, y = int(lm.x * w), int(lm.y * h)
                fingertip_points.append((x, y))
                cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

    if face_detected.multi_face_landmarks:
        for face_landmarks in face_detected.multi_face_landmarks:
            # drawing.draw_landmarks(
            #     frame,
            #     face_landmarks,
            #     mp_face_mesh.FACEMESH_LIPS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            # )

            lip_polygon_pts = []
            for idx in lip_outer_indices:
                lm = face_landmarks.landmark[idx]
                lip_polygon_pts.append((int(lm.x * w), int(lm.y * h)))

            if len(lip_polygon_pts) >= 3:
                cv.polylines(frame, [np.array(lip_polygon_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

    nail_biting = False
    if lip_polygon_pts and fingertip_points:
        contour = np.array(lip_polygon_pts, dtype=np.int32)
        for (x, y) in fingertip_points:
            distance_from_lips = cv.pointPolygonTest(contour, (x, y), True)
            if distance_from_lips > -5:
                nail_biting = True
                cv.circle(frame, (x, y), 7, (0, 0, 255), -1)

    if nail_biting:
        cv.putText(frame, "Bro not again...", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resp = requests.get(f"{BASE_URL}/led/on")
        print(resp.status_code)
        print(resp.text)
        time.sleep(5)
        resp = requests.get(f"{BASE_URL}/led/off")
        print(resp.status_code)
        print(resp.text)

    cv.imshow("Show Video", frame)
    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()