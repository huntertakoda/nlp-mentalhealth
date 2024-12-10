import pandas as pd

file_path = 'mental_health_dataset.csv'  # ensure it's in the same folder as your script
data = pd.read_csv(file_path)

# inspect the dataset

print("first 5 rows of the dataset:")
print(data.head())

print("\nsummary statistics of the dataset:")
print(data['category'].value_counts())

# text preprocessing

import re

def preprocess_text(text):
    """
    preprocesses raw text by:
    - converting to lowercase
    - removing punctuation
    - removing extra whitespace
    """
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# apply preprocessing to the dataset

data['cleaned_text'] = data['text'].apply(preprocess_text)

# inspect the first 5 rows of cleaned data

print("first 5 rows of cleaned data:")
print(data[['category', 'cleaned_text']].head())

# feature extraction

from sklearn.feature_extraction.text import TfidfVectorizer

# initialize tfidf vectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # use top 5000 words for simplicity

# fit and transform the cleaned text data

features = vectorizer.fit_transform(data['cleaned_text'])

# inspect feature matrix shape and sample features

print("feature matrix shape:", features.shape)
print("example feature names:", vectorizer.get_feature_names_out()[:10])

# train / test split

from sklearn.model_selection import train_test_split

# split data into training and testing sets

X = features  
y = data['category']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# inspect split sizes

print("training set size:", X_train.shape[0])
print("testing set size:", X_test.shape[0])

# train classifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# initialize and train logistic regression model

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# make predictions on the test set

y_pred = classifier.predict(X_test)

# evaluate the model

print("accuracy score:", accuracy_score(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))

# cross-validation to test for overfitting

from sklearn.model_selection import cross_val_score

# perform 5-fold cross-validation

cv_scores = cross_val_score(classifier, X, data['category'], cv=5, scoring='accuracy')

# inspect cross-validation results

print("cross-validation scores:", cv_scores)
print("mean accuracy:", cv_scores.mean())

# testing with noisy data

import random

def introduce_noise(text):
    """
    introduces random noise into the text:
    - random character substitutions
    - random extra spaces
    - punctuation alterations
    """
    # randomly replace a character with another
    
    if random.random() > 0.5:
        pos = random.randint(0, len(text) - 1)
        text = text[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + text[pos + 1:]

    # randomly add extra spaces
    
    if random.random() > 0.5:
        text = text.replace(' ', '  ')

    # randomly remove punctuation
    
    if random.random() > 0.5:
        text = text.replace('.', '').replace(',', '')

    return text

# create a noisy version of the dataset

data['noisy_text'] = data['cleaned_text'].apply(introduce_noise)

# inspect the first 5 rows of noisy data

print("first 5 rows of noisy data:")
print(data[['category', 'noisy_text']].head())

# feature extraction and evaluation on noisy data

# extract features from noisy data

noisy_features = vectorizer.transform(data['noisy_text'])

# split noisy features into training and testing sets

X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
    noisy_features, data['category'], test_size=0.2, random_state=42
)

# train classifier on noisy training data

classifier_noisy = LogisticRegression(max_iter=1000, random_state=42)
classifier_noisy.fit(X_train_noisy, y_train_noisy)

# evaluate the model on noisy test data

y_pred_noisy = classifier_noisy.predict(X_test_noisy)
print("accuracy on noisy data:", accuracy_score(y_test_noisy, y_pred_noisy))
print("\nclassification report on noisy data:\n", classification_report(y_test_noisy, y_pred_noisy))

# saving model and vectorizer

import pickle

# save the trained model

model_filename = "logistic_regression_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(classifier_noisy, model_file)

# save the tf-idf vectorizer

vectorizer_filename = "tfidf_vectorizer.pkl"
with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"model saved as {model_filename}")
print(f"vectorizer saved as {vectorizer_filename}")



