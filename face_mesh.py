import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

    cv2.imshow('Image', img)
    cv2.waitKey(1)