# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np


dataset = datasets.fetch_mldata("MNIST Original")


features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

count = 0
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)
    list_hog_fd.append(fd)
    count += 1
    print(count)
hog_features = np.array(list_hog_fd, 'float64')


clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)


print("done")
print("done")
print("done")
