#Once cross validate 10 folds, test on test data which includes new spoofing techniques: Test_acc -> 0.98-0.99
#Only set to train to 100 epochs
#ace=1.38

#without pca,90.08 8.62

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
train_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)
print(test_fake_wld.shape)

numRealTrain = int(pd.to_numeric(train_real_wld.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_wld.shape[0]))
numRealTest = int(pd.to_numeric(test_real_wld.shape[0]))
numFakeTest = int(pd.to_numeric(test_fake_wld.shape[0]))

####################### For wld ############################
#dimensionality reduction, PCA
pca = PCA()

#Concat real and fake training data, PCA fit it and label targets
training_wld = pd.concat([train_real_wld,train_fake_wld])
#training_wld = pca.fit_transform(training_wld)
#training_wld = pd.DataFrame(training_wld)
setTarg_train_wld = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_wld['target'] = setTarg_train_wld

#concat real and fake test data, transform it with PCA for training data, label targets
test_wld = pd.concat([test_real_wld, test_fake_wld])
#test_wld = pca.transform(test_wld)
#test_wld = pd.DataFrame(test_wld)
setTarg_test_wld = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test_wld['target'] = setTarg_test_wld

#handling some data here
numFeatures_wld = int(pd.to_numeric(training_wld.shape[1]))
feature_train_wld = training_wld.iloc[:,0:numFeatures_wld]
target_train_wld = training_wld['target']
feature_test_wld = test_wld.iloc[:,0:numFeatures_wld]
target_test_wld = test_wld['target']

feature_train_wld = feature_train_wld.values
feature_train_wld = preprocessing.scale(feature_train_wld)	#normalize
target_train_wld = target_train_wld.values
feature_test_wld = feature_test_wld.values
feature_test_wld = preprocessing.scale(feature_test_wld)
target_test_wld = target_test_wld.values


############## Start model and ML here ########################
num_ep = 100
batchSize=50
gaus_sigma=1
do=0.5

model = Sequential()
model.add(Dense(numFeatures_wld,input_dim=numFeatures_wld,init='glorot_uniform',activation='relu'))
model.add(GaussianNoise(gaus_sigma))
model.add(Dropout(do))
model.add(Dense(numFeatures_wld, activation='relu'))
model.add(Dropout(do))
#model.add(Dense(numFeatures_wld, activation='relu'))
#model.add(Dropout(do))
model.add(Dense(1,activation='sigmoid'))

#compile model
opt = Adadelta()
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

hist = model.fit(feature_train_wld, target_train_wld, nb_epoch=num_ep, batch_size=batchSize, verbose=2, validation_data=[feature_test_wld, target_test_wld], shuffle=True)

#compute ACE
#live=0 fake=1
predicted = np.round(model.predict(feature_test_wld,batch_size=50,verbose=0))
pred_target = pd.DataFrame(predicted,columns=['predicted'])
pred_target['target']=target_test_wld
live = pred_target['target'] == 0
misclas_live_rate = sum(abs( np.round((pred_target[live])['predicted']) - (pred_target[live])['target'] ) )/(pred_target[live]).shape[0]
fake = pred_target['target'] == 1
misclas_fake_rate = sum(abs( np.round((pred_target[fake])['predicted']) - (pred_target[fake])['target'] ) )/(pred_target[fake]).shape[0]
ace = (misclas_live_rate + misclas_fake_rate)/2
print ace*100
