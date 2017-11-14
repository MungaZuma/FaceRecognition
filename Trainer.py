import os

import cv2
import numpy as np
from PIL import Image

# create recognizer variable
recognizer = cv2.face.LBPHFaceRecognizer_create()

# variable for dataset
path = 'ImageDataSet'


def getImages(path):
    imagesPath = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty list of faces and ids
    faces = []
    Ids = []

    for imagePath in imagesPath:
        # open and convert image to PIL
        faceimg = Image.open(imagePath).convert('L');
        # convert PIL image into numpy array and specify conversion format
        faceNp = np.array(faceimg, 'uint8')
        # get images path and split them also convert to integer
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        # append both faces and Ids
        faces.append(faceNp)
        Ids.append(ID)
        # show the image to be trained
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    # return the values
    return Ids, faces


Ids, faces = getImages(path)
# training recognizer
recognizer.train(faces, np.array(Ids))
# save the data of the trained images.
recognizer.write('recognizer/trainingData.xml')
cv2.destroyAllWindows()
