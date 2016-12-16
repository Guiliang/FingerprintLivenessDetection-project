# Fingerprint Liveness Detection Project
## 1. Extract the features
+ Data set contains the fingerprint images from "Liveness Detection Competitions 2015"  [Available Here](http://livdet.org/registration.php).
+ Fo the image augmentation, please refer to: [augment_data.py](augment_data.py)
+ Three extractor we applied:
  + Binary Statistical Image Filter(BSIF), in folder [BSIF_feature](BSIF_feature)
  + Local Phase Quantization(LPQ), in folder [LPQ_feature](LPQ_feature)
  + Weber Local Descriptor(WLD), in folder [WLD_feature](WLD_feature)
  + Convnet-Features(CNN), in (extract_features.py)[./convnet-features/extract_features.py]
  
## 2. SVM using sklearn in python
+ Support Vector Machine(SVM) in [SVM_classifiy](SVM_classifiy)
  + [svm_fingerpint_liveness_detection_foldval.py](./SVM_classifiy/svm_fingerpint_liveness_detection_foldval.py) is for Cross Validation
  + [svm_fingerprint_liveness_detection_2015.py](./SVM_classifiy/svm_fingerprint_liveness_detection_2015.py) is for SVM test, draw the ROC curve and calculate the AUC values.
  + The result images are in [svm_2015_result](./SVM_classifiy/svm_2015_result)

## 3. Neural Network using Keras in python
+ Simple Layer Network to classify the features:
  + BSIF Features: [2015DigPer_test_BSIF.py](./NeuralNetwork/2015DigPer_test_BSIF.py)
  + LPQ Features: [2015DigPer_test_LPQ.py](./NeuralNetwork/2015DigPer_test_LPQ.py)
  + WLD Features: [2015DigPer_test_WLD.py](./NeuralNetwork/2015DigPer_test_WLD.py)
  + Mix Features: [2015DigPer_test_MixFeat.py](./NeuralNetwork/2015DigPer_test_MixFeat.py)
  + Convnet Features: [convnet_nn.py](./NeuralNetwork/convnet_nn.py)
  
+ VGG16 model are trained using [train_vgg16.py](./train_vgg16.py)
+ VGG16+vector multibranch models are trained using [train_multi.py](./train_multi.py)
+ their data generator functions can be found in [gen.py](./gen.py)

## Poster Preview
![Poster](https://cloud.githubusercontent.com/assets/22108101/21119571/016675c0-c078-11e6-80be-39bf02a124e8.png)

## The Final Report can be found [here](fingerprint-liveness-detection.pdf)
