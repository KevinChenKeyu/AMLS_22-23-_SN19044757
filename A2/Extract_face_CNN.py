import csv
import tensorflow as tf
import PIL.Image
import torch
import numpy as np
from torchvision import transforms
import os

from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([    transforms.CenterCrop((178, 178))])

global basedir, image_paths, target_size
basedir = './Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_dir = './Datasets/celeba/labels.csv'

# path to testing images
global basedir_t, image_paths_t, target_size_t
basedir_t = './Datasets/celeba_test'
images_dir_t = os.path.join(basedir_t,'img')
labels_dir_t = './Datasets/celeba_test/labels.csv'

def get_train_data():
    trainData = []
    trainLabels = []
    with open (labels_dir, "r") as labels:
        scan = csv.reader(labels, delimiter="\t")
        trainLabel = list(scan)
        for i in range(5000):
            trainLabels.append(int((int(trainLabel[i+1][3])+1)/2))
    #trainLabels = (np.array(trainLabels) + 1)/2
    #trainLabels = np.array(trainLabels, np.)
    trainLabels = torch.tensor(trainLabels).to(device)
    for j in range(5000):
        image_path = os.path.join(images_dir, "%s.jpg" % j)
        image = PIL.Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.resize((128, 128), Image.ANTIALIAS)

        image = tf.convert_to_tensor(image)
        trainData.append(image)
        #print(trainData)

    trainData = np.array(trainData, np.float32)
    #print(np.shape(trainData))
    trainData = torch.tensor(trainData)
    #print(type(trainLabels))
    #print(type(trainData))
    return trainData, trainLabels


def get_test_data():
    testData = []
    testLabels = []
    with open (labels_dir_t, "r") as labels:
        scan = csv.reader(labels, delimiter="\t")
        testLabel = list(scan)
        for i in range(1000):
            testLabels.append(int((int(testLabel[i+1][3])+1)/2))
    #testLabels = (np.array(testLabels) + 1) / 2
    #testLabels = np.array(testLabels, np.long)

    testLabels = torch.tensor(testLabels).to(device)
    for j in range(1000):
        image_path_t = os.path.join(images_dir_t, "%s.jpg" % j)
        image = PIL.Image.open(image_path_t).convert("RGB")
        image = transform(image)
        image = image.resize((128, 128), Image.ANTIALIAS)

        image = tf.convert_to_tensor(image)
        testData.append(image)
        #print(trainData)
    testData = np.array(testData, np.float32)
    #print(np.shape(trainData))
    testData = torch.tensor(testData)
    #print(type(trainLabels))
    #print(type(trainData))
    return testData, testLabels