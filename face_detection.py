import cv2
import numpy as np
print(cv2.__version__)

# overlay_img = cv2.imread('smile.png')
# if overlay_img is None:
#     print("Error: Could not load smile.png")
#     exit()

# h, w = overlay_img.shape[:2]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Couldn't load classifier")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Calculate center of the detected face
        center_x = int(x + w/2)
        center_y = int(y + h/2)
        
        # Overlay the smile image at the center of the face
        start_x = center_x - w // 2
        start_y = center_y - h // 2
        frame[start_y:start_y+h, start_x:start_x+w] = overlay_img

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
