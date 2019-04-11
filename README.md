[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/augustoicaro/SFD-CNN-TL/master?filepath=classifyAndView.ipynb)

# This code allows to reproduce results of our article:
## Seismic fault detection in real data using Transfer Learning from a Convolutional Neural Network pre-trained with synthetic seismic data

### Dependencies
	requirements.txt
	
To install all requirements in the environment use:
	pip install -r requirements.txt

### Our pre-trained model can be found in:
 	base_model/model.json
 	base_model/model.h5
 	
### Our dataset can be found in:
 	dataset/fault
 	dataset/nonfault
 
### Transfer learning methods:
 	ft.py : full fine tuning (FFT)
 	mlp.py : feature extractor with Multi Layer Perceptron (FE-MLP)
 	svm.py : feature extractor with Support Vector Machine (FE-SVM)
 
Default parameters are set to produce the results presented in the article.
 
Generated models can be saved by setting the boolean value save=true in functions create_model(). they will be save in the output/ directory.
 	
### Classification results:
 	classify.py : generates a classification file for models saved as .json and .h5
 	classify_with_SVM.py : generates a classification file for models saved as .pkl
 	
Classification files is saved in directiry classification/output/
It contains patches coordinates associated to a class value (1 for fault, 0 otherwise)
 	
We provide a region of a real section where a fault is clearly visible as demo in the classification/ directory.
Other sections can be classified modifying the classify.py and classify_with_SVM.py files.
 	
### Metrics:
 	metrics.py : computes quality metrics (accuracy, sensitivity, specificity = recall, F1-score, ROC AUC and we added precision)


