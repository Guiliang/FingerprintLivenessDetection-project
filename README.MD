# Fingerprint Liveness Detection Project

你好In this project, we explored two different methods (SVM & Neural Network) for fingerprint liveness detection.

## Feature extraction
+ We use the fingerprint images from "Liveness Detection Competitions 2015" [Available Here](http://livdet.org/registration.php).
+ For the image augmentation, please refer to: [augment_data.py](augment_data.py)
+ Four extractors we applied:
  + Binary Statistical Image Filter(BSIF), in folder [BSIF_feature](BSIF_feature)
  + Local Phase Quantization(LPQ), in folder [LPQ_feature](LPQ_feature)
  + Weber Local Descriptor(WLD), in folder [WLD_feature](WLD_feature)
  + Convnet-Features(CNN), in [extract_features.py](./convnet-features/extract_features.py)
  
## SVM using sklearn in python
+ Support Vector Machine (SVM) in [SVM_classifiy](SVM_classifiy)
  + [svm_fingerpint_liveness_detection_foldval.py](./SVM_classifiy/svm_fingerpint_liveness_detection_foldval.py) is for cross validation
  + [svm_fingerprint_liveness_detection_2015.py](./SVM_classifiy/svm_fingerprint_liveness_detection_2015.py) is for SVM test, which draws the ROC curve and calculate the AUC values.
  + [convnet_svm.py](./SVM_classifiy/convnet_svm.py) is for the CNN features
  + The result images are in [svm_2015_result](./SVM_classifiy/svm_2015_result)

## Neural Network using Keras in python
+ Simple Layer Network to classify the features:
  + BSIF Features: [2015DigPer_test_BSIF.py](./NeuralNetwork/2015DigPer_test_BSIF.py)
  + LPQ Features: [2015DigPer_test_LPQ.py](./NeuralNetwork/2015DigPer_test_LPQ.py)
  + WLD Features: [2015DigPer_test_WLD.py](./NeuralNetwork/2015DigPer_test_WLD.py)
  + Mix Features: [2015DigPer_test_MixFeat.py](./NeuralNetwork/2015DigPer_test_MixFeat.py)
  + Convnet Features: [convnet_nn.py](./NeuralNetwork/convnet_nn.py)  
+ VGG16 model is trained using [train_vgg16.py](./NeuralNetwork/train_vgg16.py)
+ VGG16+vector multibranch models are trained using [train_multi.py](./NeuralNetwork/train_multi.py). We first merge features using [merge_features.py](./NeuralNetwork/merge_features.py)
+ Their data generator functions can be found in [gen.py](./NeuralNetwork/gen.py)

## Poster Preview
![Poster](https://cloud.githubusercontent.com/assets/22108101/21119571/016675c0-c078-11e6-80be-39bf02a124e8.png)

The Final Report can be found [here](fingerprint-liveness-detection.pdf).
