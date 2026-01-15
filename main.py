"""step 1: data processing

train -> validate -> test

DIGITS:
5000 lines of digits training labels
140000 lines of digits training images
total lines_per_image = 140000/5000 = 28 lines per image

FACES:
451 lines face data training labels
31570 lines of face data train
so total lines_per_image = 31570 / 451 = 70 lines per face """

import math
import time
import random

from naive_bayes import *
from extract_data import DataLoader, extract_features

# Load Data
print("Loading data...")
loader = DataLoader()
loader.load_digits()

# Begin Classfication Algorithms
# Create and train a digit classifier
print("Training classifier...")
digit_classifier = NaiveBayes(num_classes=10)
digit_classifier.train(loader.digit_train_features, loader.digit_train_labels)

# Test on one image
print("Testing on one image...")
test_features = extract_features(loader.digit_test_images[0])
prediction = digit_classifier.predict(test_features)
actual = loader.digit_test_labels[0]

print(f"Predicted: {prediction}, Actual: {actual}")
