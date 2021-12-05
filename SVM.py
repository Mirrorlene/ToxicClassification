import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_train = np.load('dbow_Xtrain.npy')
y_train = np.load('dbow_ytrain.npy')
X_test = np.load('dbow_Xtest.npy')
y_test = np.load('dbow_ytest.npy')
X_trun = X_train[:15]
y_trun = y_train[:15]
X_test_trun = X_test
y_test_trun = y_test
print(X_trun.shape)
clf = svm.SVC()
clf.fit(X_trun, y_trun)
print('fitted')
y_predict = clf.predict(X_test_trun)
print(accuracy_score(y_test_trun, y_predict))

pca = PCA(n_components=6)
pca.fit(X_train)
X = pca.transform(X_train)
Xax = X[0:200, 0]
Yax = X[0:200, 1]
Zax = X[0:200, 2]

cdict = {0: 'red', 1: 'green'}
labl = {0: 'Non-toxic', 1: 'Toxic'}
marker = {0: '*', 1: 'o'}
alpha = {0: .3, 1: .5}

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

print(np.unique(y_train))

for l in np.unique(y_train):
    ix = np.where(y_train[0:200] == l)
    ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
                label=labl[l], marker=marker[l], alpha=alpha[l])
plt.show()
