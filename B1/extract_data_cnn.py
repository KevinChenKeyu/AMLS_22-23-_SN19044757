import csv
import tensorflow as tf
import PIL.Image
import torch
import numpy as np

import os

from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


global basedir, image_paths, target_size
basedir = './Datasets/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_dir = './Datasets/cartoon_set/labels.csv'

# path to testing images
global basedir_t, image_paths_t, target_size_t
basedir_t = './Datasets/cartoon_set_test'
images_dir_t = os.path.join(basedir_t,'img')
labels_dir_t = './Datasets/cartoon_set_test/labels.csv'

def get_train_data():
    '''
    read image files and csv file
    :return: tensor of image data and labels
    '''
    trainData = []
    trainLabels = []
    with open (labels_dir, "r") as labels:
        scan = csv.reader(labels, delimiter="\t")  # remove space in the column
        trainLabel = list(scan)
        for i in range(10000):
            trainLabels.append(int(trainLabel[i+1][2])) # select the 3rd column
    trainLabels = torch.tensor(trainLabels).to(device) # convert to tensor
    for j in range(10000):
        image_path = os.path.join(images_dir, "%s.png" % j)
        image = PIL.Image.open(image_path).convert("RGB") # reduce channel from 4 to 3
        image = image.resize((64, 64), Image.ANTIALIAS)

        image = tf.convert_to_tensor(image)
        trainData.append(image)
        #print(trainData)
    trainData = np.array(trainData, np.float32) # convert to array
    #print(np.shape(trainData))
    trainData = torch.tensor(trainData)# convert to tensor
    #print(type(trainLabels))
    #print(type(trainData))
    return trainData, trainLabels


def get_test_data(): # same function with previous one
    testData = []
    testLabels = []
    with open (labels_dir_t, "r") as labels:
        scan = csv.reader(labels, delimiter="\t")
        testLabel = list(scan)
        for i in range(2500):
            testLabels.append(int(testLabel[i+1][2]))
    testLabels = torch.tensor(testLabels).to(device)
    for j in range(2500):
        image_path_t = os.path.join(images_dir_t, "%s.png" % j)
        image = PIL.Image.open(image_path_t).convert("RGB")
        image = image.resize((64, 64), Image.ANTIALIAS)

        image = tf.convert_to_tensor(image)
        testData.append(image)
        #print(trainData)
    testData = np.array(testData, np.float32)
    #print(np.shape(trainData))
    testData = torch.tensor(testData)
    #print(type(trainLabels))
    #print(type(trainData))
    return testData, testLabels







