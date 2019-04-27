*This page is available as an executable or viewable **Jupyter Notebook**:* 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/augustoicaro/SFD-CNN-TL/master?filepath=classifyAndViewGSB.ipynb)
[![NbViwer](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjxzdmcKICAgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIgogICB4bWxuczpjYz0iaHR0cDovL2NyZWF0aXZlY29tbW9ucy5vcmcvbnMjIgogICB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4xIgogICBpZD0ic3ZnMiIKICAgaGVpZ2h0PSIyMCIKICAgd2lkdGg9IjEwOCI+CiAgPG1ldGFkYXRhCiAgICAgaWQ9Im1ldGFkYXRhMzQiPgogICAgPHJkZjpSREY+CiAgICAgIDxjYzpXb3JrCiAgICAgICAgIHJkZjphYm91dD0iIj4KICAgICAgICA8ZGM6Zm9ybWF0PmltYWdlL3N2Zyt4bWw8L2RjOmZvcm1hdD4KICAgICAgICA8ZGM6dHlwZQogICAgICAgICAgIHJkZjpyZXNvdXJjZT0iaHR0cDovL3B1cmwub3JnL2RjL2RjbWl0eXBlL1N0aWxsSW1hZ2UiIC8+CiAgICAgICAgPGRjOnRpdGxlPjwvZGM6dGl0bGU+CiAgICAgIDwvY2M6V29yaz4KICAgIDwvcmRmOlJERj4KICA8L21ldGFkYXRhPgogIDxkZWZzCiAgICAgaWQ9ImRlZnMzMiIgLz4KICA8bGluZWFyR3JhZGllbnQKICAgICB5Mj0iMTAwJSIKICAgICB4Mj0iMCIKICAgICBpZD0iYiI+CiAgICA8c3RvcAogICAgICAgaWQ9InN0b3A1IgogICAgICAgc3RvcC1vcGFjaXR5PSIuMSIKICAgICAgIHN0b3AtY29sb3I9IiNiYmIiCiAgICAgICBvZmZzZXQ9IjAiIC8+CiAgICA8c3RvcAogICAgICAgaWQ9InN0b3A3IgogICAgICAgc3RvcC1vcGFjaXR5PSIuMSIKICAgICAgIG9mZnNldD0iMSIgLz4KICA8L2xpbmVhckdyYWRpZW50PgogIDxjbGlwUGF0aAogICAgIGlkPSJhIj4KICAgIDxyZWN0CiAgICAgICBpZD0icmVjdDEwIgogICAgICAgZmlsbD0iI2ZmZiIKICAgICAgIHJ4PSIzIgogICAgICAgaGVpZ2h0PSIyMCIKICAgICAgIHdpZHRoPSIxMDgiIC8+CiAgPC9jbGlwUGF0aD4KICA8ZwogICAgIGlkPSJnMTIiCiAgICAgY2xpcC1wYXRoPSJ1cmwoI2EpIj4KICAgIDxwYXRoCiAgICAgICBpZD0icGF0aDE0IgogICAgICAgZD0iTTAgMGg0N3YyMEgweiIKICAgICAgIGZpbGw9IiM1NTUiIC8+CiAgICA8cGF0aAogICAgICAgaWQ9InBhdGgxNiIKICAgICAgIGQ9Ik00NyAwaDYxdjIwSDQ3eiIKICAgICAgIGZpbGw9IiNGMzc3MjYiIC8+CiAgICA8cGF0aAogICAgICAgaWQ9InBhdGgxOCIKICAgICAgIGQ9Ik0wIDBoMTA4djIwSDB6IgogICAgICAgZmlsbD0idXJsKCNiKSIgLz4KICA8L2c+CiAgPGcKICAgICBpZD0iZzIwIgogICAgIGZvbnQtc2l6ZT0iMTEiCiAgICAgZm9udC1mYW1pbHk9IkRlamFWdSBTYW5zLFZlcmRhbmEsR2VuZXZhLHNhbnMt)](https://nbviewer.jupyter.org/github/augustoicaro/SFD-CNN-TL/blob/master/classifyAndViewGSB.ipynb)

# This code allows to reproduce results of our article:
## Seismic fault detection in real data using Transfer Learning from a Convolutional Neural Network pre-trained with synthetic seismic data

### Notebook visualization
We provide two option to easily open and see our ipython notebooks:
- Visualize with [NbViwer](https://nbviewer.jupyter.org/github/augustoicaro/SFD-CNN-TL/blob/master/classifyAndViewGSB.ipynb)
- Visualize and modify with [Binder](https://mybinder.org/v2/gh/augustoicaro/SFD-CNN-TL/master?filepath=classifyAndViewGSB.ipynb)

We strongly recommend using NbViwer to visualize our notebooks instead open in GitHub, because you will see the interactive plots.

### Dependencies
	environment.yml
	
To install all requirements in the environment use:
	conda env create -f environment.yml

### Our F3 Block pre-trained model can be found in:
 	base_model/model.json
 	base_model/model.h5
 	
### Our F3 Block slice manualy interpreted dataset can be found in:
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
    
### Interactive examples:
	classifyAndViewF3.ipynb: Classify selected sections of F3 seismic data with all pretrained methods and show the results
	classifyAndViewGSB.ipynb: Classify selected sections of GSB seismic data with all pretrained methods and show the results
	TrainAndSave.ipynb: Train all methods with one interpreted section of a real data and save network weights
    


