import finger_until
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

methodName = ["BSIF", "LPQ", "WLD"]
methodPara = ["7_12", "3_11", "3_8"]
dataType = ["DigPerson", "CrossMatch", "GreenBit", "Hi_Scan"]
# dataType = ["DigPerson"]

for j in range(0, 4):
    tpr_list = []
    fpr_list = []
    auc_lsit = []
    for i in range(0, 3):
        train_Real = finger_until.read_lines_svm_data_format(
            "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/2015_" + methodName[
                i] + "/Data_2015_" + methodName[
                i] + "_" +
            methodPara[i] + "_motion_Train_Real_" + dataType[j] + ".txt")

        train_Spoof = finger_until.read_lines_svm_data_format(
            "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/2015_" + methodName[
                i] + "/Data_2015_" + methodName[
                i] + "_" +
            methodPara[i] + "_motion_Train_Spoof_" + dataType[j] + ".txt")

        train_data = np.concatenate((train_Real, train_Spoof), axis=0)

        test_Real = finger_until.read_lines_svm_data_format(
            "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/2015_" + methodName[
                i] + "/Data_2015_" + methodName[
                i] + "_" +
            methodPara[i] + "_motion_Test_Real_" + dataType[j] + ".txt")

        test_Spoof = finger_until.read_lines_svm_data_format(
            "/Users/liu/Desktop/CMPT-726-Machine Learning material/final project/2015_" + methodName[
                i] + "/Data_2015_" + methodName[
                i] + "_" +
            methodPara[i] + "_motion_Test_Spoof_" + dataType[j] + ".txt")

        test_data = np.concatenate((test_Real, test_Spoof), axis=0)

        train_real_length = train_Real.shape[0]
        train_spoof_length = train_Spoof.shape[0]
        test_real_length = test_Real.shape[0]
        test_spoof_length = test_Spoof.shape[0]

        train_Real_label = [1] * train_real_length
        train_Spoof_label = [0] * train_spoof_length
        train_label = train_Real_label + train_Spoof_label

        test_Real_label = [1] * test_real_length
        test_Spoof_label = [0] * test_spoof_length
        test_label = test_Real_label + test_Spoof_label

        print("train: "+str(sum(train_label))+"/"+str(len(train_label)))
        print("test: " + str(sum(test_label)) + "/" + str(len(test_label)))

        # Learn to predict each class against the other
        classifier = svm.SVC(kernel='linear', probability=True)
        y_score = classifier.fit(train_data, train_label).decision_function(test_data)
        predict_label = (classifier.predict(test_data)).tolist()

        real_acc = float(sum(predict_label[:test_real_length])) / test_real_length
        fake_acc = 1 - float((sum(predict_label[test_real_length:]))) / test_spoof_length

        real_ture = float(sum(predict_label[:test_real_length]))
        spoof_true = test_spoof_length - float(sum(predict_label[test_real_length:]))

        acc = (real_ture+spoof_true)/(test_real_length+test_spoof_length)

        Accuracy = float(np.sum(np.array(predict_label) == test_label)) / len(test_label)

        print str(methodName[i]) + "_" + str(dataType[j]) + '_testing acc is:' + str(Accuracy) + " with tpr:"+str(real_acc) + " and tnr:" +str(fake_acc)

        fpr, tpr, _ = roc_curve(test_label, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("./svm_2015_result/" + str(methodName[i]) + "_" + str(dataType[j]) + ".jpeg")
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_lsit.append(roc_auc)
        plt.close()

    plt.figure()
    for k in range(0, len(fpr_list)):
        lw = 2
        plt.plot(fpr_list[k], tpr_list[k], lw=lw,
                 label=str(methodName[k]) + "_" + dataType[j] + ' ROC curve (area = %0.2f)' % auc_lsit[k])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig("./svm_2015_result/" + str(methodName[i]) + "All_DigPerson.jpeg")
    plt.savefig("./svm_2015_result/" + str(dataType[j])+".jpeg")
    plt.close()
