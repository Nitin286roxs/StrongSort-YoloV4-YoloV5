import cv2
import numpy as np
import os
import sys

img_path = sys.argv[1]
img0 = cv2.imread(img_path)
polygon_roi = [[405,220], [1330,237], [1565,915], [275,920]]
blank = np.full((img0.shape[0], img0.shape[1], 3) , (0,255,0), np.uint8)
cv2.imwrite("red_blank.jpg", blank)
cropped_img = img0[polygon_roi[0][1]:polygon_roi[2][1],polygon_roi[0][0]:polygon_roi[2][0]]
cv2.imwrite("cropped_img.jpg", cropped_img)
blank[polygon_roi[0][1]:polygon_roi[2][1],polygon_roi[0][0]:polygon_roi[2][0]] = cropped_img
cv2.imwrite("imposed_crop.jpg", blank)
