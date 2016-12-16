'''Using PCA on 3 features extraction method. Test on 1500f
	1) Dense > Gaus(5) > Dropout(0.5) > Dense > Do(0.5) > Dense > Do(0.5), 80ep 100bs: 0.98-0.99
	2) same as (1) but remove 1 Dense and 1 do: 0.98-0.99
	3) same as (2) but do=0.3 80ep 100bs: 0.97-0.98
	4) same as (2) but last do layer has rate=0.5: 0.97-0.98
	5) Dense > Gaus(5) > Do(0.3) : 0.96-0.97 same for do=0.5
	6) Once running more than 4 dense layers, acc drop

	Run after CV
	1) 0.9744

	Ace 2.03

	without pca, 86.56 ace12.45
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, Adadelta
from keras.engine.topology import Merge

#Load features
train_real_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)
print(train_real_bsif.shape[1])

train_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)
print(train_real_lpq.shape[1])

train_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)
print(train_real_wld.shape[1])

#all features contains the same number of samples as we are extracting from identitacal samples. num of features different
numRealTrain = int(pd.to_numeric(train_real_bsif.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_bsif.shape[0]))
numRealTest = int(pd.to_numeric(test_real_bsif.shape[0]))
numFakeTest = int(pd.to_numeric(test_fake_bsif.shape[0]))

###################### For bsif ################################
#dimensionality reduction, PCA
pca = PCA()

#Concat real and fake training data, PCA fit it and label targets
training_bsif = pd.concat([train_real_bsif,train_fake_bsif])
#training_bsif = pca.fit_transform(training_bsif)
#training_bsif = pd.DataFrame(training_bsif)
setTarg_train_bsif = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_bsif['target'] = setTarg_train_bsif

#concat real and fake test data, transform it with PCA for training data, label targets
test_bsif = pd.concat([test_real_bsif, test_fake_bsif])
#test_bsif = pca.transform(test_bsif)
#test_bsif = pd.DataFrame(test_bsif)
setTarg_test_bsif = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test_bsif['target'] = setTarg_test_bsif

#Here we do not shuffle the data as in single model as we are dealing with 3 features. We have separate features for each extraction method. Shuffling will mix up the matches between each method's feature and the target labels. However, we can shuffle it during fit()

#handling some data here
numFeatures_bsif = int(pd.to_numeric(training_bsif.shape[1]))
print(numFeatures_bsif)
feature_train_bsif = training_bsif.iloc[:,0:numFeatures_bsif]
target_train_bsif = training_bsif['target']
feature_test_bsif = test_bsif.iloc[:,0:numFeatures_bsif]
target_test_bsif = test_bsif['target']

feature_train_bsif = feature_train_bsif.values
feature_train_bsif = preprocessing.scale(feature_train_bsif)	#normalize
target_train_bsif = target_train_bsif.values
feature_test_bsif = feature_test_bsif.values
feature_test_bsif = preprocessing.scale(feature_test_bsif)
target_test_bsif = target_test_bsif.values

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
print(numFeatures_lpq)
feature_train_lpq = training_lpq.iloc[:,0:numFeatures_lpq]
target_train_lpq = training_lpq['target']
feature_test_lpq = test_lpq.iloc[:,0:numFeatures_bsif]
target_test_lpq = test_lpq['target']

feature_train_lpq = feature_train_lpq.values
feature_train_lpq = preprocessing.scale(feature_train_lpq)	#normalize
target_train_lpq = target_train_lpq.values
feature_test_lpq = feature_test_lpq.values
feature_test_lpq = preprocessing.scale(feature_test_lpq)
target_test_lpq = target_test_lpq.values

########################## For wld #############################
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
print(numFeatures_wld)
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

#Start model and ML here. Here we will create a model for each different type of feature extraction method, then merge them
# Model for bsif
model_bsif = Sequential()
model_bsif.add(Dense(numFeatures_bsif,input_dim=numFeatures_bsif,init='glorot_uniform',activation='relu'))
model_bsif.add(GaussianNoise(3))
model_bsif.add(Dropout(0.3))


# Model for lpq
model_lpq = Sequential()
model_lpq.add(Dense(numFeatures_lpq,input_dim=numFeatures_lpq,init='glorot_uniform',activation='relu'))
model_lpq.add(GaussianNoise(3))
model_lpq.add(Dropout(0.3))

# Model for wld
model_wld = Sequential()
model_wld.add(Dense(numFeatures_wld,input_dim=numFeatures_wld,init='glorot_uniform',activation='relu'))
model_wld.add(GaussianNoise(3))
model_wld.add(Dropout(0.3))
model_wld.add(Dense(numFeatures_wld, activation='relu'))
model_wld.add(Dropout(0.3))

#Merge all models
model_merged = Sequential()
model_merged.add(Merge([model_bsif, model_lpq, model_wld],mode='concat',concat_axis=1))
model_merged.add(Dense(1,activation='sigmoid'))

#Show some debug output
print(model_merged.summary())

#compile model
opt = Adadelta()
model_merged.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

#fit the model
hist = model_merged.fit([feature_train_bsif, feature_train_lpq, feature_train_wld], target_train_bsif, nb_epoch=30, batch_size=50, verbose=2, validation_data=[[feature_test_bsif, feature_test_lpq, feature_test_wld], target_test_bsif], shuffle=True)
#model_merged.save_weights('2015MixFeat_DigPer_NN.h5')

#compute ACE
#live=0 fake=1
predicted = np.round(model_merged.predict([feature_test_bsif, feature_test_lpq, feature_test_wld],batch_size=50,verbose=0))
pred_target = pd.DataFrame(predicted,columns=['predicted'])
pred_target['target']=target_test_bsif
live = pred_target['target'] == 0
misclas_live_rate = sum(abs( np.round((pred_target[live])['predicted']) - (pred_target[live])['target'] ) )/(pred_target[live]).shape[0]
fake = pred_target['target'] == 1
misclas_fake_rate = sum(abs( np.round((pred_target[fake])['predicted']) - (pred_target[fake])['target'] ) )/(pred_target[fake]).shape[0]
ace = (misclas_live_rate + misclas_fake_rate)/2
print ace*100

'''
#plot some graphs
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Training loss and validation loss across epochs')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.show()


predictions = model.predict_classes(feature_test,verbose=2)
outputToText = pd.DataFrame()
outputToText['target'] = target_test
outputToText['predicted'] = predictions
outputToText.to_csv('2015_MixFeat_DigPer_NN_predict.txt')'''
