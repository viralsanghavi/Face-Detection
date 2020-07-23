import cv2
from random import randrange
from datetime import datetime, date, time, timezone


trained_face_daa = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('bg.jpeg')

webcam = cv2.VideoCapture(0)

exitText = "Long press 'Q' or 'q' to exit"
capture = "Press 'C' to capture."
font = cv2.FONT_HERSHEY_SIMPLEX
text = datetime.now()


while True:
    sfr, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame, str(exitText), (80, 20), font, 0.82, (200, 255, 255), 2, cv2.LINE_AA)
    # Add Capture feature
    #cv2.putText(frame, str(exitText), (80, 20), font, 0.82, (200, 255, 255), 2, cv2.LINE_AA)

    face_coordinates = trained_face_daa.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 10)

    cv2.imshow('Face', frame)
    cv2.waitKey(1)

    key = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()

cv2.destroyAllWindows()

print("Code Completed")
