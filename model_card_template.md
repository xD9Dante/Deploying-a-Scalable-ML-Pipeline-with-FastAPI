# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Sam Le created the model. It uses Random Forest Classifier. The parameters used a test_size=0.2 and random_state=42.

## Intended Use
This model will predict the salary of a person and whether or not that person makes > 50K or <= 50K based on a census dataset.

## Training Data
The Census Data was obtrains from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 

The Data set has 32,561 records and 15 features. One Hot Encoder was utilized on the features.

## Evaluation Data
The dataset is prepocessed and then split into a training and testing set where the size of the testing set is 20%

## Metrics
Precision: 0.7236
Recall: 0.6327
F1: 0.6751

## Ethical Considerations
There is a no biases based on the dataset provided

## Caveats and Recommendations
As this is a provided dataset, it is recommended to double check all the data to make sure there are no biases