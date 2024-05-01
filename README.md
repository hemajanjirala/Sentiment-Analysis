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

# Sentiment Analysis Model Applying.ipynb
## The following are some of the problem statement
2)Explain the role of the re (regular expression) and stopwords modules in the preprocessing step.

3)What is the purpose of the WordNetLemmatizer and how does it help in text preprocessing?

4)Why is the train_test_split function from sklearn used, and what does the random_state parameter do?

5)Explain the purpose of the LogisticRegression model and how it is trained in the given code
.
6)What does the cv.transform function do, and why is it used for both X_train and y_test?

7)What does the classification_report function provide, and why is it useful for evaluating the model's performance?

## The following are some of the conclutions.
1. The purpose of the preprocessing loop is to clean and preprocess the text data in the 'text' column of the dataframe (df).
It removes non-alphabetic characters, converts to lowercase, removes stop words, and lemmatizes the words.
The preprocessed text is stored in the 'corpus' list.

2. The 're' module is used for regular expressions to remove non-alphabetic characters from the text.
The 'stopwords' module from NLTK is used to remove common English stop words (e.g., 'the', 'a', 'is') from the text.
These steps help in reducing noise and retaining only the relevant words for further analysis.

3. The 'WordNetLemmatizer' from NLTK is used to lemmatize the words, which means converting them to their base or root form.
This helps in reducing the number of unique words and grouping together related words with the same meaning.
Lemmatization improves the accuracy of text analysis and modeling techniques.

4. The 'train_test_split' function is used to split the data into training and testing sets.
The 'test_size' parameter specifies the proportion of data to be used for testing (33% in this case).
The 'random_state' parameter ensures reproducibility by setting a fixed seed for the random number generator.

5. The 'LogisticRegression' model is a classification algorithm used for binary or multi-class classification tasks.
In this code, it is trained on the training data ('X_train_cv' and 'y_train') using the 'fit' method.
The trained model can then be used to make predictions on new, unseen data.

6. The 'cv.transform' function is likely a CountVectorizer or TfidfVectorizer method from scikit-learn.
It is used to convert the text data into a numerical format (e.g., a sparse matrix) that can be understood by machine learning models.
It is applied to both the training and testing data for consistency and compatibility with the trained model.

7. The 'classification_report' function from scikit-learn provides a detailed evaluation of the model's performance.
It calculates and displays metrics such as precision, recall, f1-score, and support for each class in the classification task.
This report is useful for understanding the model's strengths and weaknesses and identifying areas for improvement.




