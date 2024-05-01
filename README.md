# Sentiment-Analysis

## The following are some of the problem statement

1)What is the size of the dataset (number of rows and columns) after loading it from the CSV file?

2)What are the column names in the dataset?

3)What is the distribution of the target variable (sentiment) in the dataset? (e.g., how many positive and negative sentiments)

4)What is the purpose of using the PorterStemmer and removing stopwords during the text preprocessing step?

5)How many features are used in the bag-of-words model created by the CountVectorizer?

6)What is the train-test split ratio used for splitting the data into training and testing sets?

7)What is the performance of the logistic regression model on the testing data in terms of accuracy, confusion matrix, and classification report?

## The following are some of the conclutions.

1. The size of the dataset:
The dataset has 1,600,000 rows, as indicated by the filename "training.1600000.processed.noemoticon.csv".
The number of columns is 6, specified by the 'names' parameter when loading the CSV file.

2. The column names in the dataset:
The column names are: 'target', 'ids', 'date', 'flag', 'user', 'text'.

3. The distribution of the target variable (sentiment):
The distribution of the target variable (sentiment) is not shown in the provided code.
We can assume that the 'target' column contains binary values (e.g., 0 for negative sentiment and 1 for positive sentiment) based on the typical sentiment analysis task.

4. The purpose of using the PorterStemmer and removing stopwords:
The purpose of using the PorterStemmer is to reduce words to their root form (e.g., "playing" becomes "play").
Removing stopwords (common words like "the", "and", "is") helps eliminate noise and focus on the meaningful words in the text.

5. The number of features used in the bag-of-words model:
The bag-of-words model created by the CountVectorizer uses a maximum of 1,500 features.
This is specified by the 'max_features=1500' parameter.

6. The train-test split ratio:
The train-test split ratio used is 0.2, which means 20% of the data is used for testing, and 80% is used for training.
This is specified by the 'test_size=0.2' parameter in the 'train_test_split' function.

7. The performance of the logistic regression model:
The performance of the logistic regression model on the testing data is not shown in the provided code.
The code prints the accuracy, confusion matrix, and classification report, which provide insights into the model's performance.
