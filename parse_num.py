"""Contains all code for finding the recycling triangle within an image
and parsing the recycling number from the triangle"""

# import needed libaries
import imutils
import cv2
import pickle
import numpy as np
from PIL import Image

from helpers import find_similar_contours

# declare constants needed for project
thumbnail_size = (400, 400)
NUM_CNTS = 3 # number contours being detected
SVM_IM_SIZE = (28, 28)
THUMB_BORDER = .25

# read the image using cv2
filepath = "/Users/cameronwolfe/Desktop/5_ex.png"
image = cv2.imread(filepath)
image = cv2.resize(image, thumbnail_size)

# show image before detecing the shape
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert to greyscale and blur the photo
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)[1]
image_og = np.array(thresh, copy=True) # store original thresholded image to use later
thresh = (255-thresh)
cv2.imshow("Image", thresh)
cv2.waitKey(0)

# find the contours of the image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]

# find areas of all the contours found
# store them as a tuple with the contour
cnt_areas = [] 
for c in cnts:	
	area = cv2.contourArea(c)
	a_tup = (area, c)
	cnt_areas.append(a_tup)

# sort list so it's easy to find contours of same area
cnt_areas = sorted(cnt_areas, key=lambda t: t[0])

# find the contours belonging to the desired shape (i.e. recycling triangle)
shape_cnts = find_similar_contours(cnt_areas, NUM_CNTS)

# draw contours of the desired shape that was detected
cv2.drawContours(image, shape_cnts, -1, (0, 255, 0), -1)
cv2.imshow("Image", image)
cv2.waitKey(0)

# find the centroids of each of the shape components
# store x and 
centroid_xlocs = []
centroid_ylocs = []
for c in shape_cnts:
	m = cv2.moments(c)
	c_x = int(m['m10']/m['m00'])
	c_y = int(m['m01']/m['m00'])
	centroid_xlocs.append(c_x)
	centroid_ylocs.append(c_y)
	cv2.circle(image, (c_x, c_y), 2, (0, 0, 255))

# show image with the centroids displayed
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert inside of the contours to white
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(4,0))
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(-4,0))
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(0,4))
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(0,-4))


# create a thumbnail of the image of a bounding box around the centroids
thumb_im = image_og[min(centroid_ylocs): max(centroid_ylocs) + 1,
		min(centroid_xlocs): max(centroid_xlocs) + 1]
cv2.imshow("Image", thumb_im)
cv2.waitKey(0)

# center the number within the image
thresh_cnt = thumb_im.copy()
cnts = cv2.findContours(255-thresh_cnt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# find the largest contour within the current image
areas = [cv2.contourArea(c) for c in cnts]
max_index = np.argmax(areas)
cnt = cnts[max_index]
thresh_cnt = cv2.cvtColor(thumb_im.copy(), cv2.COLOR_GRAY2RGB)
cv2.drawContours(thresh_cnt, [cnt], -1, (0, 255, 0), -1)
cv2.imshow("Image", thresh_cnt)
cv2.waitKey(0)

# find hull around the object then get all associated bounding box values
hull = cv2.convexHull(cnt)
max_locs = np.amax(hull, axis=0)
min_locs = np.amin(hull, axis=0)
centered_thumb = thumb_im[min_locs[0][1]: max_locs[0][1],
			min_locs[0][0]: max_locs[0][0]]
border_px_top = int(THUMB_BORDER*centered_thumb.shape[0])
border_px_sides = int(THUMB_BORDER*centered_thumb.shape[1])
centered_thumb = cv2.copyMakeBorder(centered_thumb, border_px_top, border_px_top, border_px_sides,
		 border_px_sides, cv2.BORDER_CONSTANT, value=255)
cv2.imshow("Image", centered_thumb)
cv2.waitKey(0)

# load in the svm from the txt file
FILE_NAME = "svm.txt"
SVM_FILE = open(FILE_NAME, "rb")
svm = pickle.load(SVM_FILE)

# resize the image to a size that can be classified by the SVM
# image has already been converted to grayscale
res_im = cv2.resize(thumb_im, dsize=SVM_IM_SIZE).flatten()/255.0

# get the classification of the resized image from the SVM
num = svm.predict(res_im.reshape(1, -1))
print("Result: {0}".format(str(num)))
	


