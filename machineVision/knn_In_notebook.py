# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ImageUtils.preprocessing import SimplePreprocessor
from ImageUtils.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import pandas as pd

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())
args = {"dataset":'/Users/bartramshawd/Documents/DBS/python_code/deep_learning_personal/pyimagesearch/SB_Code/datasets/animals',
        "neighbors":1 ,
		"jobs":-1 }


# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths[0:5]

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500, label_type = "folder")
data = data.reshape((data.shape[0], 32*32*3))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),target_names=le.classes_))

# Store Predictions
predictions = model.predict(testX)
predictions[0:5]


##############################
# Dog breeds
##############################
dog_labels = pd.read_csv("/Users/bartramshawd/Documents/datasets/kaggle_dogbreed_data/labels.csv")
dog_labels.head()

test_images = '/Users/bartramshawd/Documents/datasets/kaggle_dogbreed_data/test/'
train_images = '/Users/bartramshawd/Documents/datasets/kaggle_dogbreed_data/train/'
imagePaths = list(paths.list_images(test_images))

(data, labels) = sdl.load(imagePaths, verbose=500, labels = "folder")

import os
imagePaths[0].split(os.path.sep)[-1]
