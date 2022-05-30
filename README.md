# ML-Algorithms


EEG_MEG_feature engineering.ipynb
Feature Engineering and classification using EEG and MEG time series data
 
The time,frequency domain features had to be extracted. 
Using the libraries pyeeg and MNE, features like wavelet transform, Hjorth, detrended fluctuation analysis, first order difference, Hurst exponent feature, 
petrosian fractal dimension, power, correlation dimension, bandpower and Spectral Entropy  are extracted. 
An SVM model was built with ‘balanced’ class_weight and ‘scale’ for gamma. 
The first run was using the approximate coefficients from the wavelet and the other features which crossed the easy baseline with a score of 0.9 for balanced accuracy. 
Next tsfel features from the library tsfel were added. As next step, the wavelet features were removed and the best 50 features using selectKbest were chosen. 
Since the rows are temporally coherent, the 50 features from the top 2 and bottom 2 rows were also added to each row of input features. 
The end rows of each subject were padded suitably. This improved the score to 0.947. 
Each time, the input data was split in two ways and two validation scores were computed. 
Once with the validation data being  random 25%  of the input and another way with one subject completely in validation and two subjects in training data. 
The validation score for the latter was always less than the former. 
Next, the process was repeated to the best 30 features, the hyperparameters (C and gamma) in SVM were tuned. 
The final validation in the leave one subject out case was 0.941 for balanced accuracy which was much better compared to the first run (0.9). 
