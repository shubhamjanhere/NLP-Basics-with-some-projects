"""
So in the sentiment analysis process there are a couple of stages more or less differentiated. The first is about 
processing natural language, and the second about training a model. The first stage is in charge of processing text in 
a way that, when we are ready to train our model, we already know what variables the model needs to consider as inputs. 
The model itself is in charge of learning how to determine the sentiment of a piece of text based on these variables. 

For the model part we will introduce and use linear models. They aren't the most powerful methods in terms of accuracy, 
but they are simple enough to be interpreted in their results as we will see. Linear methods allow us to define our input 
variable as a linear combination of input variables. In tis case we will introduce logistic regression.

Finally we will need some data to train our model. For this we will use data from the Kaggle competition UMICH SI650. 
"""
import pandas as pd
import numpy as np
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

# define local file names
test_data_file_name = '6_1_testdata.txt'
train_data_file_name = '6_1_training.txt'
#load files into data frames for processing.
test_data_df = pd.read_csv(test_data_file_name, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
train_data_df = pd.read_csv(train_data_file_name, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]
print("Shape of Test data-" + str(test_data_df.shape) + "\n" + "Shape of training data - " + str(train_data_df.shape))
print(train_data_df.head())
print(test_data_df.head())
print("No of labels we have for each sentiment class - \n" + str(train_data_df.Sentiment.value_counts()))
print("AVG no. of words per sentence - "+str(np.mean([len(s.split(" ")) for s in train_data_df.Text])))
"""
We will process our text sentences and create a corpus. We will also extract important words and establish them as input 
variables for our classifier. We will use basic transformations.  The requirements of a bag-of-words classifier are minimal 
. We just need to count words, so the process is reduced to do some simplification and unification of terms  and then 
count them. The simplification process mostly includes removing punctuation, lowercasing, removing stop-words, and 
reducing words to its lexical roots (i.e. stemming).
    The class sklearn.feature_extraction.text.CountVectorizer in the wonderful scikit learn Python library converts a 
collection of text documents to a matrix of token counts. This is just what we need to implement later on our bag-of-words 
linear classifier.
    First we need to init the vectoriser. We need to remove punctuations, lowercase, remove stop words, and stem words. 
All these steps can be directly performed by CountVectorizer if we pass the right parameter values. We are using porter
for stemming.
Based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html      -
"""
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
########
vectorizer = CountVectorizer(
    analyzer = 'word', tokenizer = tokenize, lowercase = True, stop_words = 'english', max_features = 85 )
corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())
corpus_data_features_nd = corpus_data_features.toarray()
print(corpus_data_features_nd.shape)
vocab = vectorizer.get_feature_names()
print(vocab)

# Sum up the counts of each vocabulary word
dist = np.sum(corpus_data_features_nd, axis=0)
# For each, print the vocabulary word and the number of times it  appears in the data set
for tag, count in zip(vocab, dist):
    print(str(count) + ':' + str(tag))
# Bag of Words Representation
print(corpus_data_features_nd)
#print(corpus_data_features)
"""
In order to perform logistic regression in Python we use LogisticRegression. But first let's split our training data in order to get an evaluation set.
"""
#from sklearn.cross_validation import train_test_split  <<<DEPRECATED>>>
from sklearn.model_selection import train_test_split

# Corpus_data_features_nd contains all of our original train and test data, so we need to exclude the unlabeled test entries
X_train, X_test, y_train, y_test  = train_test_split(
        corpus_data_features_nd[0:len(train_data_df)],
        train_data_df.Sentiment,
        train_size=0.85,
        random_state=1234)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
# Train our classifier There is a function for classification called sklearn.metrics.classification_report which calculates several
# types of (predictive) scores on a classification model.
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Finally, we can re-train our model with all the training data and use it for sentiment classification with the original (unlabeled) test set.
# train classifier
log_model = LogisticRegression()
log_model = log_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)

# get predictions
test_pred = log_model.predict(corpus_data_features_nd[len(train_data_df):])

# sample some of them
import random
spl = random.sample(range(len(test_pred)), 15)
# print text and labels
for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print(str(sentiment) + '\t' + (text))
