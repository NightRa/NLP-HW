import os
import re
import codecs

import sys

from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import cross_validation
import numpy as np


######################################################################
################### Reading the dataset ##############################
######################################################################

# read_file: file path -> IO String
def read_file(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        return f.read()


# read_file: file path -> IO String
def write_file(file_path, data):
    with codecs.open(file_path, 'w', 'utf-8') as f:
        return f.write(data)


# read_reviews: directory path -> IO List[String]
def read_reviews(reviews_dir):
    files = map(lambda file: os.path.join(reviews_dir, file), os.listdir(reviews_dir))
    # read all files
    return list(map(read_file, files))


# read_positive_reviews: directory path -> IO List[String]
def read_positive_reviews(input_dir):
    return read_reviews(os.path.join(input_dir, 'pos'))


# read_negative_reviews: directory path -> IO List[String]
def read_negative_reviews(input_dir):
    return read_reviews(os.path.join(input_dir, 'neg'))


######################################################################
################### Reading the features #############################
######################################################################

def to_words(text):
    text = re.sub(r"\r\n|\n|\t", " ", text)
    text = re.sub(r" +", " ", text)
    return text.split(" ")


def read_features(file):
    return to_words(read_file(file))


######################################################################
################### Feature Vectors for Step 1 #######################
######################################################################

def contains(words, keyword):
    for word in words:
        if keyword in word:
            return 1
    return 0


def calc_feature_vector(feature_model, text):
    words = to_words(text)
    # We set 1 for a feature word if it's contained in a word in the text.
    # Our features are roots of words.
    return list(map(lambda keyword: contains(words, keyword), feature_model))


######################################################################
##################### Read The Data ##################################
######################################################################

input_dir = sys.argv[1]
words_file_input_path = sys.argv[2]
best_words_file_output_path = sys.argv[3]
positives = read_positive_reviews(input_dir)
negatives = read_negative_reviews(input_dir)
texts = positives + negatives
target = np.array(([1] * len(positives)) + ([0] * len(negatives)))


######################################################################
################### Learning #########################################
######################################################################

def evaluateClassifier(data, target, clf, name):
    scores = cross_validation.cross_val_score(clf, data, target, cv=StratifiedKFold(target, 10, shuffle=True))
    print("- %s: %.2f (+- %.2f)" % (name, scores.mean(), (scores.std() * 2)))


# Evaluate the features with the classifier transformer clfF
def evaluateF(data, target, clfF):
    print()
    evaluateClassifier(data, target, clfF(svm.SVC()), "SVM")
    evaluateClassifier(data, target, clfF(MultinomialNB()), "Naive Bayes")
    evaluateClassifier(data, target, clfF(tree.DecisionTreeClassifier()), "DecisionTree")
    evaluateClassifier(data, target, clfF(neighbors.KNeighborsClassifier()), "KNN")
    print()


def evaluate(data, target):
    evaluateF(data, target, lambda clf: clf)


def evaluateFeatures(features):
    positive_feature_vectors = list(map(lambda text: calc_feature_vector(features, text), positives))
    negative_feature_vectors = list(map(lambda text: calc_feature_vector(features, text), negatives))

    data = positive_feature_vectors + negative_feature_vectors

    evaluate(data, target)


######################################################################
################### Manually Selected Features #######################
######################################################################


def step1():
    print("step1 (manually generated features):")
    features = read_features(words_file_input_path)
    evaluateFeatures(features)


######################################################################
###################### Bag Of Words ##################################
######################################################################

def step2():
    def text_clf(clf):
        return Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf),
                         ])

    print("step2 (bag-of-words):")
    evaluateF(texts, target, text_clf)
    # The amount of distincts words in the Bag of Words = 22878
    # print(len(CountVectorizer(stop_words='english').fit(texts).get_feature_names()))


######################################################################
####################### SelectKBest ##################################
######################################################################

def selectKBestFeatures():
    counts = CountVectorizer(stop_words='english').fit(texts, target)
    counted = counts.transform(texts)
    tfidfed = TfidfTransformer().fit_transform(counted, target)
    k_best = SelectKBest(k=50).fit(tfidfed, target)
    k_best_support = k_best.get_support()
    # Take all the indices which were chosen, and map them back to the feature names - the words.
    selected_features = list(
        map(lambda i: counts.get_feature_names()[i],
            filter(lambda i: k_best_support[i],
                   range(len(k_best_support)))))

    return selected_features


def step3():
    k_best_features = selectKBestFeatures()
    write_file(best_words_file_output_path, "\r\n".join(k_best_features))
    return k_best_features


######################################################################
################# Classify with selected features ####################
######################################################################

def step4(best_k_features):
    def text_clf(clf):
        return Pipeline([('vect', CountVectorizer(stop_words='english', vocabulary=best_k_features)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf),
                         ])

    print("step4 (selected best features):")
    evaluateF(texts, target, text_clf)


######################################################################
###################### Run Everything ################################
######################################################################

step1()
step2()
k_best_features = step3()
step4(k_best_features)
