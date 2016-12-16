'''Input of network will be BSIF extracted features(4096) and they are from Digital Persona tech. year 2015
   Use PCA for feature selection, as we observed validation accuracy does not decrease, PCA could help? Yes it helps!
   Feed these features into a layer network

	1) Dense(relu) > Gauss(1) > Dense(relu), 80 epochs 100bs: 0.95-0.96
	2) Dense(relu) > Gauss(1) > Dropout(0.5) > Dense(sigmoid), 80 epochs, 100bs:0.97
	3) Dense(relu) > Gauss(5) > Dropout(0.3) > Dense(sigmoid), 100epochs 100bs: 0.97-0.98
	4) Dense(relu) > Gauss(5) > Dropout(0.3) > Dense(relu) > Dropout(0.3) > Dense(sigmoid), 100epochs 100bs:0.98-0.99
	5) Use same config as (4) but with dropout 0.5, and add Dense(relu) > Dropout(0.5), 80ep 100bs:0.99-1

	Test on CV
	1) finetuning result, test_acc = 0.9828
	2) ace=1.45
	3) without pca test_acc 86.08, ace=12.95
	

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, Adadelta

#Load features
train_real = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)
numRealTrain = int(pd.to_numeric(train_real.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake.shape[0]))
numRealTest = int(pd.to_numeric(test_real.shape[0]))
numFakeTest = int(pd.to_numeric(test_fake.shape[0]))

#PCA dimensionality reduction
pca = PCA()

#Concat real and fake training data, PCA fit it and label targets
training = pd.concat([train_real,train_fake])
#training = pca.fit_transform(training)
#training = pd.DataFrame(training)
setTarg_train = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training['target'] = setTarg_train

#concat real and fake test data, transform it with PCA for training data, label targets
test = pd.concat([test_real, test_fake])
#test = pca.transform(test)
#test = pd.DataFrame(test)
setTarg_test = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test['target'] = setTarg_test

#Shuffle randomly to prevent any patterns or overfit
training = shuffle(training)
training.reset_index(inplace=True)
test = shuffle(test)
test.reset_index(inplace=True)

#num of features after PCA
numFeatures = int(pd.to_numeric(training.shape[1]))

#combine, extract this and that, some data handling
feature_train = training.iloc[:,0:numFeatures]
target_train = training['target']
feature_test = test.iloc[:,0:numFeatures]
target_test = test['target']

feature_train = feature_train.values
feature_train = preprocessing.scale(feature_train)	#normalize
target_train = target_train.values
feature_test = feature_test.values
feature_test = preprocessing.scale(feature_test)
target_test = target_test.values

#Start model and ML here
# Model 1
model = Sequential()
model.add(Dense(numFeatures,input_dim=numFeatures,init='glorot_uniform',activation='relu'))
model.add(GaussianNoise(3))
model.add(Dropout(0.3))
#model.add(Dense(numFeatures, activation='relu'))
#model.add(Dropout(0.3))
#model.add(Dense(numFeatures, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

# Show some debug output
print (model.summary())

#compile model
opt = Adadelta()
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

#fit the model
hist = model.fit(feature_train, target_train, nb_epoch=30, batch_size=50, verbose=2, validation_data=[feature_test, target_test])
#model.save_weights('2015BSIF_PCA_DigPer_NN.h5')

#compute ACE
#live=0 fake=1
predicted = np.round(model.predict(feature_test,batch_size=50,verbose=0))
pred_target = pd.DataFrame(predicted,columns=['predicted'])
pred_target['target']=target_test
live = pred_target['target'] == 0
misclas_live_rate = sum(abs( np.round((pred_target[live])['predicted']) - (pred_target[live])['target'] ) )/(pred_target[live]).shape[0]
fake = pred_target['target'] == 1
misclas_fake_rate = sum(abs( np.round((pred_target[fake])['predicted']) - (pred_target[fake])['target'] ) )/(pred_target[fake]).shape[0]
ace = (misclas_live_rate + misclas_fake_rate)/2
print ace*100
#plot loss graph
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title('Training loss and validation loss across epochs')
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend(['train','test'])
#plt.show()

"""predictions = model.predict_classes(feature_test,verbose=2)
outputToText = pd.DataFrame()
outputToText['target'] = target_test
outputToText['predicted'] = predictions
outputToText.to_csv('2015_BSIF_PCA_DigPer_NN_predict.txt')"""
