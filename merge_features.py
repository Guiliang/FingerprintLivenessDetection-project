# Creates merged one-dimensional feature matrix from Matlab feature outputs.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing

numFeatures=2000 # Desired number of resulting features; used to make static models for multi-branch.

#Change directories as needed
for directory in ['/local-scratch/alice/cmpt726/LivDet2015/Training/Digital_Persona',
        '/local-scratch/alice/cmpt726/LivDet2015/Testing/Digital_Persona']:
    b0 = pd.read_csv(directory + "/Data_2015_BSIF_7_12_motion_Spoof_DigPerson.txt", sep='\s+', header=None)
    b1 = pd.read_csv(directory + "/Data_2015_BSIF_7_12_motion_Real_DigPerson.txt", sep='\s+', header=None)
    numb0 = int(b0.shape[0])
    numb1 = int(b1.shape[0])

    #Optional: Do PCA
    bb = pd.concat([b0,b1])
    pca = PCA(n_components=numFeatures)
    bb = pca.fit_transform(bb)

    #Output
    bb = pd.DataFrame(bb)
    bb = bb.values
    bb = preprocessing.scale(bb)  # normalize
    np.savetxt(directory + '/bsif.txt', bb)


