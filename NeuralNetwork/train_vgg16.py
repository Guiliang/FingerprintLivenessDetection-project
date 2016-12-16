# aya43@sfu.ca; yataoz@sfu.ca; last modified 20161209

# Train vgg16 model

import numpy as np
from gen import image_generator
from model import create_vgg16
import keras.backend as K
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from utils import LearnRateScheduler, WeightsWriter 
from keras.callbacks import EarlyStopping
import cPickle as pickle

train_dir ='/local-scratch/alice/cmpt726/LivDet2015/Training/Digital_Persona'
test_dir ='/local-scratch/alice/cmpt726/LivDet2015/Testing/Digital_Persona'
path = 'cnn_only3'
best_weights_file = path + '/best_weights.pkl'
final_weights_file = path + '/final_weights.pkl'
history_file = path + '/history.pkl'

imsize = (224, 224)  # all images will be resized to 224x224 vgg16 input
learning_rate = .001
momentum = 0.9
early_stop_patience = 3
lr_step_decay = 0.9
lr_decay_patience = 2
num_epcs = 25
do = .5

#make model
model = create_vgg16(imsize, do)
ad = Adadelta()
sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=True, clipnorm=5.)
model.compile(loss='binary_crossentropy', optimizer=ad, metrics=['accuracy'])

wwriter = WeightsWriter(filepath=best_weights_file, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
early_stop = EarlyStopping(monitor='val_acc', patience=early_stop_patience, verbose=0, mode='auto')
lr_schd = LearnRateScheduler(lr_step_decay, monitor='val_acc', patience=lr_decay_patience, verbose=0, mode='auto')

#train
hist = model.fit_generator(
        image_generator(train_dir),
        samples_per_epoch=2000,
        nb_epoch=num_epcs,
        validation_data = image_generator(test_dir, range(1000)),
        nb_val_samples=1000,
        callbacks=[early_stop, lr_schd, wwriter])

#output
weights = model.get_weights()
with open(final_weights_file, 'wb') as f:
    pickle.dump(weights, f)
try:
    hist.history['lr']=K.get_value(hist.model.optimizer.lr)
    with open(history_file, 'w') as f:
        pickle.dump(hist.history, f)
except Exception as e:
    print e
print "Finish"
