# aya43@sfu.ca; last modified 20161209
# 'bb' is one-dimensional vector matrix, 'tt' is the target vector

# Generators for training neural network models

import os
import re
import numpy as np
import pandas as pd
import random
#from sklearn.decomposition import PCA
import cv2
#from sklearn import preprocessing

#read image only
def image_generator(directory, sample_indices=None, rand=True, imsize=(224, 224), ext='jpg|jpeg|bmp|png', one='/Live/'):
	# t=os.getcwd()
	# os.chdir('/local-scratch/alice/cmpt726')
	paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn if
	         os.path.isfile(os.path.join(dp, f)) and
	         re.match('([\w]+\.(?:' + ext + '))', f)]
	# os.chdir(t)
	assert len(paths), "no image has been read..."
	paths.sort()
	ind = [i for i in range(len(paths))]  # random image
	if rand:
		random.seed(100)
		random.shuffle(ind)
	if sample_indices is not None:
		ind = [ind[i] for i in sample_indices]
	while True:
		for i in ind:
			path = paths[i]
			x1 = cv2.imread(path)
			x1 = cv2.resize(x1, imsize)
			x1 = np.transpose(x1, axes=(2, 0, 1))
			x1 = np.expand_dims(x1, axis=0)
			t = np.zeros((1,))
			if one in path:
				t[0] = 1
			yield x1, t


#read vector+image
def multi_generator(directory, sample_indices=None, imsize=(224, 224), ext='jpg|jpeg|bmp|png', one='/Live/', filename="/bsif.txt"):
	paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn if
	         os.path.isfile(os.path.join(dp, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]
	assert len(paths), "no image has been read..."
	pathsl = [p for p in paths if one in p]
	pathsp = [p for p in paths if one not in p]
	pathsl.sort()
	pathsp.sort()
	paths = pathsp + pathsl

	bb = pd.read_csv(directory + filename, sep='\s+', header=None)
	numb0 = len(pathsp)
	numb1 = len(pathsl)
	t0 = [0] * numb0
	t1 = [1] * numb1
	tt = t0 + t1

	ind = [i for i in range(len(paths))]  # random image
	random.seed(100)
	random.shuffle(ind)
	if sample_indices is not None:
		ind = [ind[i] for i in sample_indices]

	while True:
		for i in ind:
			path = paths[i]
			x1 = cv2.imread(path)
			x1 = cv2.resize(x1, imsize)
			x1 = np.transpose(x1, axes=(2, 0, 1))
			x1 = np.expand_dims(x1, axis=0)
			x2 = np.expand_dims(bb[i], axis=0)
			t = np.zeros((1,))
			t[0] = tt[i]
			yield {'img': x1, 'bsif': x2}, t


#read vector
def vec_generator(directory, sample_indices=None, numFeatures = 2000, filename="/bsif.txt"):
	bb = pd.read_csv(directory + filename, sep='\s+', header=None)
	numb0 = 1000
	numb1 = 1000
	if 'Testing' in directory:
		numb0 = 1500
	bb = pd.DataFrame(bb)
	bb = bb.values
	t0 = [0] * numb0
	t1 = [1] * numb1
	tt = t0 + t1

	ind = [i for i in range(np.shape(np.array(bb))[0])]  # random image
	random.seed(100)
	random.shuffle(ind)
	if sample_indices is not None:
		ind = [ind[i] for i in sample_indices]

	while True:
		for i in ind:
			x2 = np.expand_dims(bb[i], axis=0)
			t = np.zeros((1,))
			t[0] = tt[i]
			yield x2, t


if __name__ == '__main__':
	train_dir = '/local-scratch/alice/cmpt726/LivDet2015/bb/Digital_Persona'
	multi_generator(train_dir)
