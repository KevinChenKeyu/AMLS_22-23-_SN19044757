import os
import numpy as np
from keras_preprocessing import image
import cv2

import dlib

# PATH TO TRAINING IMAGES
global basedir, image_paths, target_size
basedir = './Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

# path to testing images
global basedir_t, image_paths_t, target_size_t
basedir_t = './Datasets/celeba_test'
images_dir_t = os.path.join(basedir_t,'img')
labels_filename_t = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    gender_labels = {}
    lines = labels_file.readlines()
    checker1 = lines[5][-6] # -
    checker2 = lines[5][-3:] # -1
    checker3 = lines[1][-2:] # 1
    for index,line in enumerate(lines):
        if line[-3:] == checker2 and line[-6] == checker1: # -1 -1
            gender_labels.update({index-1: -1})
        if line[-3:] == checker2 and line[-6] != checker1: # 1 -1
            gender_labels.update({index-1: 1})
        if line[-2:] == checker3 and line[-5] == checker1: # -1  1
            gender_labels.update({index-1: -1})
        if line[-2:] == checker3 and line[-5] != checker1 and line[-6] != checker1: # 1 1
            gender_labels.update({index-1: 1})
    #print(gender_labels)
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name = int(img_path.split("\\")[-1].split('.')[0])
            #print(file_name)
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    print(np.shape(landmark_features))
    return landmark_features, gender_labels


def extract_features_labels_test():
    image_paths_t = [os.path.join(images_dir_t, x) for x in os.listdir(images_dir_t)]
    target_size_t = None
    labels_file_t = open(os.path.join(basedir_t, labels_filename_t), 'r')
    genderlabels_t = {}
    lines_t = labels_file_t.readlines()
    checker1 = lines_t[5][-6]  # -
    checker2 = lines_t[5][-3:]  # -1
    checker3 = lines_t[1][-2:]  # 1
    for index, line in enumerate(lines_t):
        if line[-3:] == checker2 and line[-6] == checker1:  # -1 -1
            genderlabels_t.update({index - 1: -1})
        if line[-3:] == checker2 and line[-6] != checker1:  # 1 -1
            genderlabels_t.update({index - 1: 1})
        if line[-2:] == checker3 and line[-5] == checker1:  # -1  1
            genderlabels_t.update({index - 1: -1})
        if line[-2:] == checker3 and line[-5] != checker1 and line[-6] != checker1:  # 1 1
            genderlabels_t.update({index - 1: 1})

    if os.path.isdir(images_dir_t):
        allfeatures = []
        alllabels = []
        for img_path in image_paths_t:
            file_name = int(img_path.split("\\")[-1].split('.')[0])
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size_t,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                allfeatures.append(features)
                alllabels.append(genderlabels_t[file_name])

    landmark_features = np.array(allfeatures)
    gender_labels = (np.array(alllabels) + 1) / 2  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels
extract_features_labels()

