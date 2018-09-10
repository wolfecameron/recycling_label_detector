"""Implements an SVM using the sklearn library and trains the svm to recognize digits - this
svm is then serialized and used to recognize digits on the recycling symbols
"""

import pickle
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from mnist import MNIST

# all constants
NUM_EX = 15000
SIDE_LEN = 28
TEST_SIZE = 1500

# connect mnist data reader to the file where mnist is located
mndata = MNIST('/Users/cameronwolfe/Desktop/coding_files/mnist')
images, labels = mndata.load_training()
images_t, labels_t = mndata.load_testing()

# convert images into a numpy arrays
images = np.array([np.array(im) for im in images])
labels = np.array(labels)
images_t = np.array([np.array(im) for im in images_t])
labels_t = np.array(labels_t)

# define a list of numbers that will be classified
good_labels = [1, 2, 3]

# eliminate all 0s from the dataset, not needed for this
# numbers above 6 also do not exist for recycling
bad_indices = np.where((labels == 0) | (labels == 4))
bad_indices_t = np.where((labels_t == 0) | (labels_t == 4))
images = np.delete(images, bad_indices, axis=0)
labels = np.delete(labels, bad_indices)
images_t = np.delete(images_t, bad_indices_t, axis=0)
labels_t = np.delete(labels_t, bad_indices_t)
bad_indices = np.where(labels > 5)
bad_indices_t = np.where(labels_t > 5)
images = np.delete(images, bad_indices, axis=0)
labels = np.delete(labels, bad_indices)
images_t = np.delete(images_t, bad_indices_t, axis=0)
labels_t = np.delete(labels_t, bad_indices_t)

# anything 3 or greater can be labeled as the same category
labels[labels > 3] = 3
labels_t[labels_t > 3] = 3

# cut the training data into a smaller subset 
# reduces training time
# balance all the classes
for l in good_labels:
	bad_indices = np.where(labels == l)[0][int(NUM_EX/len(good_labels)): ]
	images = np.delete(images, bad_indices, axis=0)
	labels = np.delete(labels, bad_indices)

# limit the size of the test set
images_t = images_t[: TEST_SIZE]
labels_t = labels_t[: TEST_SIZE]

# invert the images
images = (255 - images)/255.0
images_t = (255 - images_t)/255.0

# split into training and testing data - mnist already comes split so not needed
#TEST_RATIO = .3
#x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_RATIO)

# instantiate the SVM
GAMMA = .05
C_PARAM = 5
classifier = svm.SVC(C=C_PARAM, gamma=GAMMA)

# run the training and fit classifier to the digit labels
classifier.fit(images, labels)

# check to see how the classifier generalizes on the test data
result = classifier.predict(images_t)
print("Printing Confusion Matrix/Results on Test Data for SVM")
print(metrics.confusion_matrix(labels_t, result))

# serialize the resulting model with pickle to be used in the recycling project
FILE_NAME = "svm.txt"
SVM_FILE = open(FILE_NAME, "wb")
pickle.dump(classifier, SVM_FILE)
print("SVM classifier written to svm.txt file.")
