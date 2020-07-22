import cv2
from random import randrange


trained_face_daa = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('bg.jpeg')

webcam = cv2.VideoCapture(0)


while True:
    sfr, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_daa.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 10)

    cv2.imshow('Face', frame)
    cv2.waitKey(1)

    key = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()

cv2.d
cv2.destroyAllWindows()

print("Code Completed")
