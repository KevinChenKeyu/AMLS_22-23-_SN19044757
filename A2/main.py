import cv

import time
from PIL import Image
import numpy as np
import csv



def Cut_mouth(pic):
    """
    crop the mouth of the image for emotion detection
    :param pic: the image ready to be cut
    :return: the image of mouth
    """
    cutted = img[pic[0][1]:pic[0][1] + pic[0][3], pic[0][0]:pic[0][0]+pic[0][2]]
    return cutted


