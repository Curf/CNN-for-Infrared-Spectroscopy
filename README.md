# README:
---
## Deep Learning and Chemometrics: Quantitative and Qualitative Spectroscopy Interpretation of Aqueous Solutions
##### Colin Murphy; CS 615 Deep Learning - Spring 2020
---
[Download copy of the final paper for analysis details](CS615_Project__CM.pdf)

#### Packages Needed:
 - NumPy, Matplotlib, Pandas, Keras, Sklern, TensorFlow (backend)

#### Code for cleaning of the original .txt files (provided by the International Diffuse Reflectance Conference) is performed in "NIR_DATA_CLEAN.py"; however, you do not need to execute the script, as the cleaned csv datasets have been included in this distribution (within data folder).

#### All further data processing (standardization, data augmenting, and one-hot encoding) can be found in "DATA_PREP_MODELING.py" 

#### The modeling structure (built using Keras and TensorFlow) can also be found within "DATA_PREP_MODELING.py"

<br>

## CODE NOTES:
##### "GlobalStand" function performs standardization over the entire dataset that is given (zero-mean and unit standard deviation). 
##### "load_dataset" function loads the csv data file into a pandas dataframe - indexed by the sample ID and applies "GlobalStand" to spectroscopy dataset.
##### "plot_spectrum" function plots a given spectrum (requires dataframe format) for visualization. 
##### "one_hot" converts analyte names to unique ID keys. 
##### "data_augment" function create a random augmented spectrum from a given sample.
##### "model_in_data" function performs the final data prep before training - indicating "one_hot_y" to be applied if data is being used for classification.  Furthermore, it performs data augmentation on the training dataset, creating 10 augmented samples, per original sample and concatenating them.    
 

#### "classification_model" defines the structure of the CNN network for classification, with the KerasClassifier wrapper being used to feed the model into the sklearn cross_validation function to perform a stratified 5-fold cross-validation and return the mean and standard deviation of the accuracy.
#### "regression_model" defines the structure of the CNN network for regression, with the training being performed when calling the function.  Function returns a tuple of the trained model and the testing dataset error.


##### "coeff_determination" function calculates the coefficient of determination (R^2) between true and predicted.
##### "huber" function calculates the huber error between true and predicted.
##### "rise" function calculates the rise error between true and predicted.

##### "f_map_visual" function graphs the feature maps of the trained model's first convolution layer for visualization.

<br>

#####

