import cv2

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceCascade = cv2.CascadeClassifier('opencv-3.4/data/haarcascades/haarcascade_eye.xml')
#faceCascade = cv2.CascadeClassifier('Pasek/classifier/cascade.xml')

while True:
    ret, frame = video_capture.read()
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        centerx = int(x+(w/2))
        centery = int(y+(h/2))
        cv2.circle(frame, (centerx, centery), 5, (0, 0, 0), -1)
        print(centerx)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
