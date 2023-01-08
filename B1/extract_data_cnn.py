import time
import numpy as np
import cv2
import os
import random
from keras.preprocessing import image
from PIL import Image



global basedir, image_paths, target_size
basedir = '../Datasets/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

# path to testing images
global basedir_t, image_paths_t, target_size_t
basedir_t = '../Datasets/cartoon_set_test'
images_dir_t = os.path.join(basedir_t,'img')
labels_filename_t = 'labels.csv'

def get_train_data():
    trainData = []
    trainLabel = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    edited = labels_file.read().replace('\t', ',')
    labels_file.close()
    labels_file = open(os.path.join(basedir, labels_filename), 'w')
    labels_file.write(edited)
    labels_file.close()
    train_labels = open(os.path.join(basedir, labels_filename))
    lines = train_labels.readlines()
    for line in lines[1:501]:
        trainLabel.append(int(line.split(',')[2]))


    for img_path in image_paths[:500]:

        #print(file_name)
        # load image

        img = cv2.imread(img_path)


        trainData.append(img/500)

    trainLabel = np.array(trainLabel, np.float32)
    trainData = np.array(trainData, np.float32)

    return trainData, trainLabel





#image_paths = [os.path.join(images_dir1, l) for l in os.listdir(images_dir1)]
#target_size = None
#labels_file = open(os.path.join(basedir1, labels_filename), 'r')
#temp = labels_file.read().replace('\t', ',')

#print(temp[1:])
#labels_file.close()
#labels_file = open(os.path.join(basedir1, labels_filename), 'w')
#labels_file.write(temp)
#labels_file.close()
#train_labels = open(os.path.join(basedir1, labels_filename))

#print(train_labels)
#lines = train_labels.readlines()
#train_gender = {int(line.split(',')[1].split('.')[0]):int(line.split(',')[2]) for line in lines[1:5001]}