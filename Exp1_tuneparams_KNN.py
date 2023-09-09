import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from __future__ import print_function, division
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from PIL import Image
import glob
import skimage.io as io
import time
import seaborn as sns
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import copy


# Load images

characters = ['Luigi', 'Bowser']
ch_map ={ 'Luigi': 0, 'Bowser': 1}

imgs = {} 
labels = {}



for folder in ['train', 'val']:
	imgs[folder] = [] 
	labels[folder] = []
	for ch in characters:
		path = '../data/images/Exp1_TuneParams'
		dirr = os.path.join(path, folder ,ch, '*.jpg')
		for im_path in glob.glob(dirr):
			img = Image.open(im_path)
			arr = np.array(img)
			arr2= arr.reshape([arr.shape[0]*arr.shape[1]*arr.shape[2],1])
			imgs[folder].append(arr2)
			labels[folder].append(ch)


X_train = np.concatenate(imgs['train'], axis = 1).T
X_test  = np.concatenate(imgs['val'], axis = 1).T
y_train = np.array([ch_map[ch] for ch in labels['train']])
y_test  = np.array([ch_map[ch] for ch in labels['val']])


########################################################################
## Tune KNN 
########################################################################

K = range(1,30)
error_knn = []

for k in K:
	clf = KNeighborsClassifier(k).fit(X_train, y_train)
	y_test_pred = clf.predict(X_test)
	error_knn.append(1 - accuracy_score(y_test, y_test_pred))

print(k)
print(error_knn)
print(min(error_knn))

plt.figure()
plt.title('K-Nearest Neighbor: Test Error vs. Num_neighbors')
plt.xlabel('k')
plt.ylabel('Test Error')
plt.plot(K, error_knn)
plt.show()















# names = ["Nearest Neighbors"
# 		, "Logistic Regression"
# 		, "SVM"
# 		, "Kernel SVM"]

# classifiers = [KNeighborsClassifier(3)
# 	, LogisticRegression(max_iter = 1000)
# 	, SVC(kernel='linear')
# 	, SVC(kernel='rbf')]


# accuracy = {}
# confusion_mat = {}
# f1  = {}
# recall = {}

# for name, clf in zip(names, classifiers):

# 	# Fit model

# 	print(name)
# 	t0 = time.time()
# 	clf.fit(X_train, y_train)
# 	y_test_pred = clf.predict(X_test)
# 	t1 = time.time()
# 	print("Run time =", t1-t0)

# 	# Confusion matrix

# 	confusion_mat[name] = confusion_matrix(y_test, y_test_pred)
# 	print(confusion_mat[name])
# 	plt.figure()
# 	sns.heatmap(confusion_mat[name], annot=True, fmt='.0f')
# 	plt.title(name)


# 	# Scores

# 	report  = classification_report(y_test, y_test_pred)
# 	accuracy[name] =  classification_report(y_test, y_test_pred, output_dict=True)['weighted avg']['precision']
# 	f1[name] = classification_report(y_test, y_test_pred, output_dict=True)['weighted avg']['f1-score']
# 	recall[name] = classification_report(y_test, y_test_pred, output_dict=True)['weighted avg']['recall']

# 	print(report)

# #plt.show()
