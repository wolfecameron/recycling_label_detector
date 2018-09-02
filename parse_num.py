"""Contains all code for finding the recycling triangle within an image
and parsing the recycling number from the triangle"""

# import needed libaries
import imutils
import cv2
import numpy as np
import pytesseract as tes_ocr
from PIL import Image

from helpers import find_similar_contours

# declare constants needed for project
thumbnail_size = (400, 400)
NUM_CNTS = 3 # number contours being detected

# read the image using cv2
filepath = "/Users/cameronwolfe/Desktop/3_ex.jpg"
image = cv2.imread(filepath)
image = cv2.resize(image, thumbnail_size)
image_og = np.array(image, copy=True) # create copy of original pixels

# show image before detecing the shape
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert to greyscale and blur the photo
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
thresh = (255-thresh)

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
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(2,0))
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(-2,0))
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(0,2))
cv2.drawContours(image_og, shape_cnts, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(0,-2))



# eliminate border of recycling shape contour from the image
for rec_cnt in shape_cnts:
	for px_loc in rec_cnt:
		# change each pixel value within the contours to white
		r_loc = px_loc[0][0]
		c_loc = px_loc[0][1]
		image_og[c_loc, r_loc] = (255, 255, 255)

# create a thumbnail of the image of a bounding box around the centroids
thumb_im = image_og[min(centroid_ylocs): max(centroid_ylocs) + 1,
		min(centroid_xlocs): max(centroid_xlocs) + 1]
cv2.imshow("Thumb", thumb_im)
cv2.waitKey(0)


result = tes_ocr.image_to_string(Image.fromarray(thumb_im))
print("Result: " + str(result))





	


