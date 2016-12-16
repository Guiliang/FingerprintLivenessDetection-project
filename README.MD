# Fingerprint Liveness Detection Project
## 1. Extract the features
+ Data set contains the fingerprint images from "Liveness Detection Competitions 2015"  [Available Here](http://livdet.org/registration.php).
+ Three extractor we applied:
  + Binary Statistical Image Filter(BSIF), in folder [BSIF_feature](BSIF_feature)
  + Local Phase Quantization(LPQ), in folder [LPQ_feature](LPQ_feature)
  + Weber Local Descriptor(WLD), in folder [WLD_feature](WLD_feature)
  
## 2. SVM using sklearn in python
+ Support Vector Machine(SVM) in [SVM_classifiy](SVM_classifiy)
  + [svm_fingerpint_liveness_detection_foldval.py](./SVM_classifiy/svm_fingerpint_liveness_detection_foldval.py) is for Cross Validation
  + [svm_fingerprint_liveness_detection_2015.py](./SVM_classifiy/svm_fingerprint_liveness_detection_2015.py) is for SVM test, draw the ROC curve and calculate the AUC values.
  + The result images are in [svm_2015_result](./SVM_classifiy/svm_2015_result)

## 3. Neural Network using Keras in python
+ VGG16 model are trained using [train_vgg16.py](./train_vgg16.py)
+ VGG16+vector multibranch models are trained using [train_multi.py](./train_multi.py)
+ their data generator functions can be found in [gen.py](./gen.py)

## Poster Preview
![Poster](https://cloud.githubusercontent.com/assets/22108101/21119571/016675c0-c078-11e6-80be-39bf02a124e8.png)

## The Final Report can be found [here](fingerprint-liveness-detection.pdf)
