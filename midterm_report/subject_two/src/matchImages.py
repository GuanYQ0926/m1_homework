from matplotlib import pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np


img1Path = '../asset/img1.jpg'
img2Path = '../asset/img2.jpg'
img1 = cv2.imread(img1Path, 0)
img2 = cv2.imread(img2Path, 0)
