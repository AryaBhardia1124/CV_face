import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
