import cv2
import sys
import numpy as np
import json


# Get user supplied values
imagePath = sys.argv[1]
cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
facedimenstions = [];

for (x, y, w, h) in faces:
    facedimenstions.append({'x': x , 'y' : y, 'w': w , "h" : h });

sys.stdout.write(json.dumps(facedimenstions))
cv2.waitKey(0)