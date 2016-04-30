import os
import re
import codecs

from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
import numpy as np


######################################################################
################### Reading the dataset ##############################
######################################################################

# read_file: file path -> IO String
def read_file(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        return f.read()


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
################### To words #########################################
######################################################################

def to_words(text):
    text = re.sub(r"\r\n|\n|\t", " ", text)
    text = re.sub(r" +", " ", text)
    return text.split(" ")


######################################################################
################### Reading the features #############################
######################################################################

def read_features():
    return to_words(read_file(os.path.join(root_dir, 'root_features.txt')))


######################################################################
################### Feature Vectors ##################################
######################################################################

def contains(words, keyword):
    for word in words:
        if keyword in word:
            return 1
    return 0


def get_feature_model(positives, negatives):
    return read_features()


def calc_feature_vector(feature_model, text):
    words = to_words(text)
    # We set 1 for a feature word if it's contained in a word in the text.
    # Our features are roots of words.
    return list(map(lambda keyword: contains(words, keyword), feature_model))


def prettify_feature_vector(feature_vector, features):
    return zip(features, feature_vector)


def contained_keywords(feature_vector, features):
    list = []
    for keyword, contained in zip(features, feature_vector):
        if contained:
            list.append(keyword)
    return list


######################################################################
###################### Testing #######################################
######################################################################
clf = svm.SVC()
# clf = MultinomialNB()

def features_subset(feature_vector, bitset):
    new_vector = []
    for i in range(len(bitset)):
        if bitset[i] == '1':
            new_vector.append(feature_vector[i])
    return new_vector


def calc_accuracy(bitset):
    positive_vectors = list(map(lambda vector: features_subset(vector, bitset), positive_feature_vectors))
    negative_vectors = list(map(lambda vector: features_subset(vector, bitset), negative_feature_vectors))
    data = positive_vectors + negative_vectors
    scores = cross_validation.cross_val_score(clf, data, target, cv=4)
    return scores.mean()

root_dir = "C:\\Users\\Ilan\\Programming\\University\\NLP-HW\\hw3\\"

def get_features():
    input_dir = os.path.join(root_dir, "imdb1.train\\")
    positives = read_positive_reviews(input_dir)
    negatives = read_negative_reviews(input_dir)
    features = get_feature_model(positives, negatives)

    positive_feature_vectors = list(map(lambda text: calc_feature_vector(features, text), positives))
    negative_feature_vectors = list(map(lambda text: calc_feature_vector(features, text), negatives))

    data = positive_feature_vectors + negative_feature_vectors
    target = np.array(([1] * 1000) + ([0] * 1000))
    return data, target, positive_feature_vectors, negative_feature_vectors, features

if __name__ == '__main__':

    data, target, positive_feature_vectors, negative_feature_vectors, features = get_features()

    ######################################################################
    ################### Learning #########################################
    ######################################################################

    while True:
        bitset = input()
        accuracy = calc_accuracy(bitset)
        print(accuracy)
