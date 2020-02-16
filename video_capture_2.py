from cv2 import cv2
import time

face_cascade = cv2.CascadeClassifier("url-to-this-file\\haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame = video.read()
    if check:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.flip(frame, 1)
        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize= (30, 30)
            )
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ends the video window when 'Esc' key is pressed  
            break
video.release()
cv2.destroyAllWindows()