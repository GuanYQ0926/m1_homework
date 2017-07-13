from matplotlib import pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np


class ImgStitching:

    def __init__(self, img1Path='../asset/img1.jpg',
                 img2Path='../asset/img2.jpg'):
        self.img1Path = img1Path
        self.img2Path = img2Path
        self.img1 = cv2.imread(img1Path, 0)
        self.img2 = cv2.imread(img2Path, 0)
        self.dispaly_img1 = cv2.imread(img1Path)
        self.dispaly_img2 = cv2.imread(img2Path)

    def findFeatures(self):
        # step1: find features
        orb = cv2.ORB()
        self.kp1, self.des1 = orb.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = orb.detectAndCompute(self.img2, None)

    def matchFeatures(self):
        # step2: match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matches = bf.match(self.des1, self.des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        # match => [Dmatch]
        # Dmatch => queryIdx.pt:kp1,  trainIdx.pt:kp2

    def calcHomography(self):
        # step3: calculate Homography
        selected_pts1 = []
        selected_pts2 = []
        for i in xrange(4):
            kp1_index = self.matches[1 + i].queryIdx
            kp2_index = self.matches[1 + i].trainIdx
            temp_pt1 = [self.kp1[kp1_index].pt[0],
                        self.kp1[kp1_index].pt[1]]
            temp_pt2 = [self.kp2[kp2_index].pt[0],
                        self.kp2[kp2_index].pt[1]]
            selected_pts1.append(temp_pt1)
            selected_pts2.append(temp_pt2)
        self.img1_pts = np.array(selected_pts1)  # .reshape(-1, 1, 2)
        self.img2_pts = np.array(selected_pts2)  # .reshape(-1, 1, 2)
        print(self.img1_pts, self.img2_pts)
        self.h, self.status = cv2.findHomography(
            self.img1_pts, self.img2_pts, cv2.RANSAC, 5.0)
        print(self.h)

    def drawFeatures(self):
        for i in range(4):
            cv2.circle(self.dispaly_img1, (int(self.img1_pts[i][0]), int(
                self.img1_pts[i][1])), 5, (0, 255, 255), -1)
            cv2.circle(self.dispaly_img2, (int(self.img2_pts[i][0]), int(
                self.img2_pts[i][1])), 5, (0, 255, 255), -1)
        cv2.imshow('img1', self.dispaly_img1)
        cv2.imshow('img2', self.dispaly_img2)
        c = cv2.waitKey(0)
        if 'q' == chr(c & 255):
            cv2.destroyAllWindows()

    def processImage(self):
        img_out = cv2.warpPerspective(self.img1, self.h * -1000,
                                      (2 * self.img2.shape[1], 2
                                       * self.img2.shape[0]))
        img_out[:self.img2.shape[0], :self.img2.shape[1]] = self.img2
        # plt.figure(1)
        # plt.imshow(self.img1)
        # plt.figure(2)
        # plt.imshow(self.img2)
        # plt.figure(3)
        # plt.imshow(img_out)
        # plt.show()
        cv2.imshow('img1', self.img1)
        cv2.imshow('img2', self.img2)
        cv2.imshow('result', img_out)
        c = cv2.waitKey(0)
        if 'q' == chr(c & 255):
            cv2.destroyAllWindows()

    def testHomography(self):
        for point in self.img1_pts:
            temp_pt = np.array([point[0][0], point[0][1], 1])
            result = temp_pt.dot(self.h)
            result = result / result[2] * (-1000.0)
            print(temp_pt, result)


if __name__ == '__main__':
    test = ImgStitching()
    test.findFeatures()
    test.matchFeatures()
    test.calcHomography()
    # test.drawFeatures()
    test.processImage()
