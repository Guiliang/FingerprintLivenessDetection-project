# imported utilities

from keras.callbacks import Callback
import keras.backend as K
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import warnings
import pdb

class LearnRateScheduler(Callback):
    def __init__(self, step_decay, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        super(LearnRateScheduler, self).__init__()
        assert step_decay<1, 'step_decay must be less than 1.'
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.step_decay = step_decay

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('LearnRateScheduler mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('LearnRateScheduler requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: decay learning rate.' % (epoch))

                assert hasattr(self.model.optimizer, 'lr'), \
                    'Optimizer must have a "lr" attribute.'
                lr = K.get_value(self.model.optimizer.lr)
                lr *= self.step_decay
                #assert type(lr) == float, 'The output of the "schedule" function should be float.'
                lr=float(lr)
                K.set_value(self.model.optimizer.lr, lr)

            self.wait += 1

class WeightsWriter(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto'):

        super(WeightsWriter, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('WeightsWriter mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    weights=self.model.get_weights()
                    with open(filepath, 'wb') as f:
                        pickle.dump(weights, f)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            weights=self.model.get_weights()
            with open(filepath, 'wb') as f:
                pickle.dump(weights, f)

