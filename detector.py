import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
##cretate new recognizer
rec = cv2.face.LBPHFaceRecognizer_create();
##load training data
rec.read('recognizer/trainingData.xml')
# rec.train(faces, np.array(Ids));
##create id place holder
id = 0
##font for display
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, config = rec.predict(gray[y:y + h, x:x + w]);
        if (id == 1):
            id = "Saha"
        elif (id == 2):
            id = "Zuma"
        elif (id == 3):
            id = "Mramba"
        elif (id == 4):
            id = "Masinde"
        elif (id == 5):
            id = "Eduu"
        else:
            id = "Unknown"
        cv2.putText(img, str(id), (x, y + h), 1,font, (0,0,255));
        # print 'the face is of:', id
    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xff == ord('k')
    if k == 10:
        break

cap.imshow()
cap.release()
cv2.destroyAllWindows()
