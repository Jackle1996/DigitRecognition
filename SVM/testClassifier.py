# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import os
print(os.getcwd())

# Load the classifier
clf = joblib.load(r"SVM\digits_cls.pkl")

# Read the input image 
im = cv2.imread(r"TestData\testpic.PNG")
cv2.imshow("Raw image", im)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#cv2.imshow("cvtcolor", im_gray)

im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
#cv2.imshow("GaussianBlur", im_gray)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#im_th = cv2.bitwise_not(im_th)
cv2.imshow("treshhold", im_th)

# Find contours in the image
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # rect[2]=width, rect[3]=height
    if rect[2] < 20 or rect[3] < 20 or rect[2] > rect[3]:
        continue
    # Draw the rectangles    
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    #if pt1 > 0:
     #   pt1 = pt1 * -1
    #if pt2 > 0:
     #   pt2 = pt2 * -1

    if roi.size == 0:
        continue
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, (3 ,3))

    # Dilate = die zahl wird schmaler gemacht, linien werden dünner
    roi_erode = cv2.erode(roi, (3, 3), 1)
    roi_dilate = cv2.dilate(roi, (3, 3), 1)
    
    # Calculate the HOG features
    if True:
        # hog(bild, )
        roi_hog_fd = hog(roi, orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)
        roi_hog_fd_erode = hog(roi_erode,  orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)
        roi_hog_fd_dilate = hog(roi_dilate,  orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)

        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        nbr_erode = clf.predict(np.array([roi_hog_fd_erode], 'float64'))
        nbr_dilate = clf.predict(np.array([roi_hog_fd_dilate], 'float64'))

        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)
        cv2.putText(im, " "+str(int(nbr_erode[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        cv2.putText(im, "  "+str(int(nbr_dilate[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        print("Detected number: ", nbr)
        print("Detected erode number: ", nbr_erode)
        print("Detected dilated number: ", nbr_dilate)
        cv2.imshow("treshhold", roi)
        cv2.imshow("treshhold", roi_erode)
        cv2.imshow("treshhold", roi_dilate)
        #cv2.waitKey()

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()