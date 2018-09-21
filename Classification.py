from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
from PIL import Image


def plotImageNumber(index):
    some_digit = X[index]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def plotImage(image_array):
    some_digit_image = image_array.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


def myCross_val_score(n_splits):
    skfolds = StratifiedKFold(n_splits=n_splits, random_state=42)
    score = []
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train[test_index])
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        score.append(n_correct / len(y_pred))
    return score


mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
sgd_clf = SGDClassifier(random_state=42, max_iter=10)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
sgd_clf.fit(X_train, y_train)

# im = Image.open("newPIC0.png").convert("L").resize((28, 28))
# im.save("newScaled.png")
# img = list(im.getdata())
# img_array = np.asarray(img)
# img_array = img_array.reshape(28, 28)
# # print(img_array)
# # print(img_array.reshape(1, -1))
# print("I guess the HandWrite Number is: " + str(sgd_clf.predict(img_array.reshape(1, -1))[0]))
# plotImage(img_array)

plotImageNumber(344)
