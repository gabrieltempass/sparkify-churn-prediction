# Sparkify Churn Prediction

A model to predict churn for a music streaming company, with Spark running on an AWS EMR cluster.

## Description

The project analyzes the characteristics and behaviors of the users from a music streaming service and, based on that, predicts which ones will churn.

The dataset used is a log file containing all the user interactions with the platform over time, such as: songs listened, pages visited and profile information.

The Sparkify project is divided into the following tasks:

### 1. Load and Clean Dataset

Load the dataset and prepare it to be analyzed, through: ajusting the data types, dealing with invalid or missing data, correcting the encoding and parsing the [user agent](https://en.wikipedia.org/wiki/User_agent) column.

### 2. Exploratory Data Analysis

Perform an exploratory data analysis for the whole user log file, viewing the number of unique users, the period of the available data and plotting the numerical and categorical features. Also define that the `Churn` will be the `Cancellation Confirmation` events, which happen for both paid and free users, and create a new column for it, to be used as the label for the model.

### 3. Feature Engineering

Build out the features that seem promising to train the model on and put them in a users features matrix. Then, plot the features against the target label (Churn), to indentify and understand possible correlations and causalities.

### 4. Modeling

Create a machine learning model to predict which users will churn based on their characteristics and behaviors. Making a pipeline and processing the users features matrix to be in the correct input format for the model, with indexer, one hot encoder and vector assembler. Tune the parameters through cross validation with grid search, show the results and save the best model.

### 5. Conclusion

Discuss which methods to use moving forward, and how to test how well the recommendations are working for engaging users.

## Dependencies

To run locally:
- Python 3.7.9
- Java 1.8.0_271
- PySpark 2.4.4
- Jupyter notebook 6.2.0
- NumPy 1.19.2
- Pandas 1.2.0
- Matplotlib 3.3.2
- ua-parser 0.10.0
- user-agents 2.2.0

To run on an AWS EMR cluster:
- PySpark 2.4.4
- NumPy 1.14.5
- Pandas 1.0.0
- ua-parser 0.10.0
- user-agents 2.2.0

## Execute

To run locally:

1. Clone the git repository:

`git clone https://github.com/gabrieltempass/article-recommendation-engine.git`

2. Go to the project's directory.
3. Open the Jupyter notebook, with the command:

`jupyter notebook "notebook/sparkify_churn_prediction.ipynb"`

To run on an AWS EMR cluster:

## Notebook

The Jupyter notebook contains the five parts detailed in the README description.

## Dataset

There are two datasets: the mini, to be used in the local mode to explore the data and test the model, and the full, to be used in a distributed cluster (simulating a production environment). Both of them were provided by Udacity and are available at AWS S3 with these filepaths:

- Full dataset (12 Gb, around 26 million rows):
  
  `s3n://udacity-dsnd/sparkify/sparkify_event_data.json`

- Mini dataset (128 Mb, around 260 thousand rows, 1% of the full dataset):
  
  `s3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json`