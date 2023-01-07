import time
import numpy as np
import cv2
import os
import random


trainData = []
trainLabel = []

for subdir, dirs, files in os.walk("./assets/mwc/Refined/"):
    #shuffle  array for better training
    random.shuffle(files)
    for file in files:
        filepath = subdir + os.sep + file
        img = cv2.imread(filepath)
        trainData.append(img/255)#normalisation
        trainLabel.append(1 if file.startswith('men') else 0)
        print(file)