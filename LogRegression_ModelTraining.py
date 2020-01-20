__author__ = "Ashley Melanson"
__version__ = "1.0.1"

import csv
import pandas

import nltk
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import pickle

### GLOBAL VARIABLES ###########################################################

IN_FILENAME =
OUTFILENAME =
MODEL_FILENAME = 'final_model_pickle.sav'

sentiment_classes = {
    0 : negative,
    1 : somewhat negative,
    2 : neutral,
    3 : somewhat positive,
    4 : positive
}

tokenizer = RegexpTokenizer('\s+', gaps=True)

# I want to look at unigrams AND bigrams
# we want to ignore words that appear in more than 80% of the documents (will eliminate insignificant words)
# we want to ignore words that appear in less than 5% of the documents (i.e. names)
tfv = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=0.05, tokenizer=tokenizer)

logr = LogisticRegression()

## TF_IDF CALCULATIONS #########################################################

def compute_tf(doc, word):
    return doc.count(word) / len(doc)

def compute_idf(word, docList):
    N = len(docList)
    N_docs_with_word = 0
    for doc in docList:
        if word in doc:
            N_docs_with_word += 1
    return log(N / float(N_docs_with_word))

def compute_tfidf(doc, docList):
    tfidf_scores = {}
    tfidf_scores = {word : compute_tf(doc, word) * compute_idf(word, docList) for word in doc}
    return tfidf_scores

## DATA PRE-PROCESSING #########################################################

dtrain = []
labels = []
with open(IN_FILENAME, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    sample_data = list(reader)

for dct in sample_data:
    compute_tfidf(dct['responses'], sample_data)

# clean up

## FEATURE EXTRACTION #########################################################

# Build the tf_idf features
tf_train_features = tfv.fit_transform(dtrain['responses'])

# how many words extracted by the vectorizer
len(tfv.get_feature_names())

dense = tf_train_features.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

# x_train is the training dataset
# y_train is the labels for the data in x_train
# x_test is the test dataset
# y_test are the labels for the data in x_test
X_train, X_test, y_train, y_test = train_test_split(
    tf_train_features,
    labels,
    train_size=0.80,
    random_state=42
) # what is random_state? seed?

# Train the logistic regression model to this dataset
logr_model = logr.fit_(X=X_train, y=y_train)
# use logistic regression to add predictions on the test set
y_pred = logr_model.predict(X_test)

# Model Evaluations
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

# Save the trained model
# Take note of python version, library versions,
pickle.dump(logr_model, open(MODEL_FILENAME, 'wb'))


### TO LOAD IN ANOTHER file
trained_logr_model = pickle.load(open(MODEL_FILENAME, 'rb'))
result = trained_logr_model.score(X_test, Y_test) # will return accuracy on unseen data
