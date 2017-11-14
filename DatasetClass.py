import numpy as np
import cv2

# specify haarcascade file that is to be used
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# specify the source of the video feed
cap = cv2.VideoCapture(0)

# create id for the captured images
identity = raw_input("Enter user ID = ")
# set image counter limit
counterNum = 0

# loop for all the activities within the video feed4

while True:
    ret, img = cap.read()
    # convert feed to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # method to detect multiple faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # loop for drawing around the detected faces
    for (x, y, w, h) in faces:
        # counter
        counterNum = counterNum + 1
        # write saved image into images folder
        cv2.imwrite("ImageDataSet/User." + str(identity) + "." + str(counterNum) + ".jpg", gray[y:y + h, x:x + w])
        # draw circle around detected face
        # img = cv2.circle(img, (x + 40, y + 40), 100, (0, 255, 0), 2)
        # draw rectangle around detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # delay
        cv2.waitKey(100);

    # show feed
    cv2.imshow('Face detected', img);
    cv2.waitKey(1);
    if (counterNum > 15):
        break

cap.release()
cv2.destroyAllWindows()
