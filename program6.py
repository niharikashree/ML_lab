from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Step 1: Load specific categories from the 20 Newsgroups dataset
print("Fetching training and test data...", flush=True)
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
print("Data fetched successfully.", flush=True)

# Step 2: Explore dataset
print("Number of training documents:", len(twenty_train.data))
print("Number of test documents:", len(twenty_test.data))
print("Target categories:", twenty_train.target_names)
print("\nSample training document:")
print("\n".join(twenty_train.data[0].split("\n")[:10]))  # print first 10 lines of first document
print("Target category of sample document:", twenty_train.target[0])

# Step 3: Convert text data to term frequency matrix
print("\nVectorizing text data using CountVectorizer...", flush=True)
count_vect = CountVectorizer()
X_train_tf = count_vect.fit_transform(twenty_train.data)

# Step 4: Convert TF matrix to TF-IDF
print("Transforming to TF-IDF representation...", flush=True)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
print("TF-IDF matrix shape:", X_train_tfidf.shape)

# Step 5: Train a Multinomial Naive Bayes classifier
print("Training MultinomialNB classifier...", flush=True)
model = MultinomialNB()
model.fit(X_train_tfidf, twenty_train.target)

# Step 6: Preprocess test data
print("Transforming test data...", flush=True)
X_test_tf = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)

# Step 7: Make predictions
print("Making predictions...", flush=True)
predicted = model.predict(X_test_tfidf)

# Step 8: Evaluation
print("\n=== Evaluation Results ===")
print("Accuracy:", accuracy_score(twenty_test.target, predicted))
print("\nClassification Report:\n", classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print("Confusion Matrix:\n", confusion_matrix(twenty_test.target, predicted))
