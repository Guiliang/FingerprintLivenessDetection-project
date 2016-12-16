from sklearn.model_selection import StratifiedKFold

import finger_until
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm
from sklearn.metrics import roc_curve, auc

train_Real = finger_until.read_lines_svm_data_format(
    "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/BSIF/BSIF_7_12_motion_Train_Real_Bio.txt")

train_Spoof = finger_until.read_lines_svm_data_format(
    "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/BSIF/BSIF_7_12_motion_Train_Spoof_Bio.txt")

train_data = np.concatenate((train_Real, train_Spoof), axis=0)

test_Real = finger_until.read_lines_svm_data_format(
    "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/BSIF/BSIF_7_12_motion_Test_Real_Bio.txt")

test_Spoof = finger_until.read_lines_svm_data_format(
    "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/BSIF/BSIF_7_12_motion_Test_Spoof_Bio.txt")

test_data = np.concatenate((test_Real, test_Spoof), axis=0)

train_Real_label = np.array([1] * 1000)
train_Spoof_label = np.array([0] * 1000)
train_label = np.concatenate([train_Real_label, train_Spoof_label])
test_label = train_label

fig_num = 0
kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
# kernels = ["rbf", "sigmoid", "precomputed"]
for kernels_select in kernels:
    fig_num += 1
    # classifier = svm.SVC(kernel=kernels_select, probability=True, degree=3, gamma='auto')
    classifier = svm.SVC(kernel=kernels_select, probability=True)
    cv = StratifiedKFold(n_splits=6)
    tmp = cv.split(train_data, test_data)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2
    i = 0

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    plt.figure(fig_num)
    for (train, test), color in zip(cv.split(train_data, train_label), colors):
        print((test.shape)[0])
        probas_ = classifier.fit(train_data[train], train_label[train]).predict_proba(test_data[test])
        predict_label = (classifier.predict(test_data[test])).tolist()
        print("acc is :" + str(1- float(sum(map(abs, predict_label - test_label[test]))) / (test.shape)[0]))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(test_label[test], probas_[:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(train_data, train_label)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(kernels_select + ".jpeg")

# kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
#
# for kernel_select in kernels:
#
#     cv = StratifiedKFold(n_splits=6)
#     # Learn to predict each class against the other
#     classifier = svm.SVC(kernel=kernel_select, probability=True, degree=3)
#     y_score = classifier.fit(train_data, train_label).decision_function(test_data)
#     predict_label = (classifier.predict(test_data)).tolist()
#
#     acc = (float(sum(predict_label[:1000])) / 1000 + (1 - float((sum(predict_label[1000:])) / 1000))) / 2
#
#     print('testing acc is:', str(acc))
#
#     # # Compute ROC curve and ROC area for each class
#
#     fpr, tpr, _ = roc_curve(test_label, y_score)
#     roc_auc = auc(fpr, tpr)
#
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
