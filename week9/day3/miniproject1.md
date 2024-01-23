# Mini Project : Sentiment Analysis with Random Forest Classifier

## Project Overview
In this mini project, we will perform sentiment analysis on a dataset using a Random Forest Classifier. 
Sentiment analysis involves determining the sentiment or emotion expressed in a text, such as positive, negative, or neutral. 
We will train a machine learning model to predict sentiment based on text data.

## What you will learn

- Data preprocessing techniques for text data.
- Training a machine learning model for sentiment analysis.
- Hyperparameter tuning using GridSearchCV, RandomSearchCV, and Bayesian optimization.
- Comparing model performance across different hyperparameter tuning methods.
- Model evaluation using various metrics.


## Preparation

- Have a notebook prepared


## Dataset 

We will use the dataset which is available on Kaggle for sentiment analysis, which consists of a sentence and its respective sentiment as a target variable. 
This dataset contains 3 separate files named train.txt, test.txt and val.txt.
You can find the dataset [here](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

## Task

- import the necessary libraries.
- Read the training data and validation data, create the data frame with read_csv().
- Check for the various target labels using seaborn
- Preprocess the data to get rid of any characters apart from alphabets, convert the string to lowercase, check for stopwords in the data and get rid of them and perform lemmatization on each word.
- Use Random Forest Classifier to train your data.
- Tune the model hyperparameters using GridSearchCV, RandomSearchCV, and Bayesian optimization. Document the configurations and results of each method.
- Read and preprocess the test data similarly to the training data.
- Evaluate the model on the test data using metrics like Accuracy Score, Precision Score, Recall Score, and Confusion Matrix. Create a ROC curve for visual evaluation.
- Compare the performance of the model with different hyperparameter tuning methods and determine the most effective approach.

to learn more about which hyperparameters tuning method to pick : https://towardsdatascience.com/how-to-optimize-hyperparameters-of-machine-learning-models-98baec703593

### Submit your Daily Challenge :

Don’t forget to push to GitHub
