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


Siamese_network_Image_classification.ipynb

Image classification based on taste of food in the picture

Some of the important libraries used:
Library -  Keras,tensorflow,PIL,sklearn

After referring to several researches, a siamese network model built on a pre trained nets like VGG that learnt using triplet loss seemed very promising. The maximum dimensions of the images in the data set were read off (342x512 pixels). Each image was preprocessed to reach this target size with suitable padding on all four sides. A data generator was built to read each line of the training triplets and save the first image as anchor, second as positive and third as negative - since in the training data set the second image is of a food item that is always closer in taste to the food item in the first image (hence positive) than to the food item in the third image (hence negative). The initial batch size for the data generator was set as 16 (reads 16 rows of triplets).

A deep neural net was built wherein each image started with a VGG16 layer  and ‘imagenet’ weights. This was further continued with a Dense layer, followed by global max pooling layer. Finally the transformed features for all three images were concatenated. The Euclidean distances between these transformed features of the anchor image to positive and anchor image to negative were used to define the triplet loss. The weights from the trained model were saved and used to predict the label for the test data set, based on the triplet loss. If the Euclidean distance of the transformed features of the first image (A) to that of the second image (B) was less than the distance between the transformed features of the first image(A) to the third image(C), the image in A was classified to be closer in taste to B(label =  1) and label = 0 otherwise. Due to computational constraints, the batch_size could be increased only if the validation data generator was first removed. Increasing batch_size to 64 improved the score. Also, reducing the image size did not affect the performance very much and hence the image size was reduced to 225*225 _pixels to increase batch_size to 100. Any further increase in batch_size did not improve performance of the model. The parameters (number of epochs, optimizer, batch_normlaization,dropouts) were tuned in the model by several test runs and the final model produced a score of 0.661. Since further modifications to the neural net or parameters, did not improve the performance,this was decided as the final model.
