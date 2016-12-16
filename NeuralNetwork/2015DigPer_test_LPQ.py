#Once cross validate 10 folds, test on test data which includes new spoofing techniques: Test_acc -> 0.99-1
#Only set to train to 100 epochs

#ace = 1.17

#without pca, test acc 94.32, ace 4.92
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, Adadelta
from keras.engine.topology import Merge

#Load features
train_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)
print(test_fake_lpq.shape)

numRealTrain = int(pd.to_numeric(train_real_lpq.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_lpq.shape[0]))
numRealTest = int(pd.to_numeric(test_real_lpq.shape[0]))
numFakeTest = int(pd.to_numeric(test_fake_lpq.shape[0]))

####################### For lpq ############################
#dimensionality reduction, PCA
pca = PCA()

#Concat real and fake training data, PCA fit it and label targets
training_lpq = pd.concat([train_real_lpq,train_fake_lpq])
#training_lpq = pca.fit_transform(training_lpq)
#training_lpq = pd.DataFrame(training_lpq)
setTarg_train_lpq = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_lpq['target'] = setTarg_train_lpq

#concat real and fake test data, transform it with PCA for training data, label targets
test_lpq = pd.concat([test_real_lpq, test_fake_lpq])
#test_lpq = pca.transform(test_lpq)
#test_lpq = pd.DataFrame(test_lpq)
setTarg_test_lpq = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test_lpq['target'] = setTarg_test_lpq

#handling some data here
numFeatures_lpq = int(pd.to_numeric(training_lpq.shape[1]))
feature_train_lpq = training_lpq.iloc[:,0:numFeatures_lpq]
target_train_lpq = training_lpq['target']
feature_test_lpq = test_lpq.iloc[:,0:numFeatures_lpq]
target_test_lpq = test_lpq['target']

feature_train_lpq = feature_train_lpq.values
feature_train_lpq = preprocessing.scale(feature_train_lpq)	#normalize
target_train_lpq = target_train_lpq.values
feature_test_lpq = feature_test_lpq.values
feature_test_lpq = preprocessing.scale(feature_test_lpq)
target_test_lpq = target_test_lpq.values


############## Start model and ML here ########################
num_ep = 100
batchSize=50
gaus_sigma=1
do=0.3

model = Sequential()
model.add(Dense(numFeatures_lpq,input_dim=numFeatures_lpq,init='glorot_uniform',activation='relu'))
model.add(GaussianNoise(gaus_sigma))
model.add(Dropout(do))
#model.add(Dense(numFeatures_lpq, activation='relu'))
#model.add(Dropout(do))
#model.add(Dense(numFeatures_lpq, activation='relu'))
#model.add(Dropout(do))
model.add(Dense(1,activation='sigmoid'))

#compile model
opt = Adadelta()
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

hist = model.fit(feature_train_lpq, target_train_lpq, nb_epoch=num_ep, batch_size=batchSize, verbose=2, validation_data=[feature_test_lpq, target_test_lpq], shuffle=True)

#compute ACE
#live=0 fake=1
predicted = np.round(model.predict(feature_test_lpq,batch_size=50,verbose=0))
pred_target = pd.DataFrame(predicted,columns=['predicted'])
pred_target['target']=target_test_lpq
live = pred_target['target'] == 0
misclas_live_rate = sum(abs( np.round((pred_target[live])['predicted']) - (pred_target[live])['target'] ) )/(pred_target[live]).shape[0]
fake = pred_target['target'] == 1
misclas_fake_rate = sum(abs( np.round((pred_target[fake])['predicted']) - (pred_target[fake])['target'] ) )/(pred_target[fake]).shape[0]
ace = (misclas_live_rate + misclas_fake_rate)/2
print ace*100

