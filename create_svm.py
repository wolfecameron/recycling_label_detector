"""Implements an SVM using the sklearn library and trains the svm to recognize digits - this
svm is then serialized and used to recognize digits on the recycling symbols
"""

import pickle
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# import the digit images
dig_data = datasets.load_digits()

# zip the images and their associated labels into a list
# flatten images for SVM
images = [im.flatten() for im in dig_data.images]
labels = list(dig_data.target)

# split into training and testing data
TEST_RATIO = .3
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_RATIO)

# instantiate the SVM
GAMMA = .001
classifier = svm.SVC(gamma=GAMMA)

# run the training and fit classifier to the digit labels
classifier.fit(x_train, y_train)

# check to see how the classifier generalizes on the test data
result = classifier.predict(x_test)
print("Printing Confusion Matrix/Results on Test Data for SVM")
print(metrics.confusion_matrix(y_test, result))

# serialize the resulting model with pickle to be used in the recycling project
FILE_NAME = "svm.txt"
SVM_FILE = open(FILE_NAME, "wb")
pickle.dump(classifier, SVM_FILE)
