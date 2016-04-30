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
root_dir = "C:\\Users\\Ilan\\Programming\\University\\NLP-HW\\hw3\\"
input_dir = os.path.join(root_dir, "imdb1.train\\")
positives = read_positive_reviews(input_dir)
negatives = read_negative_reviews(input_dir)
features = get_feature_model(positives, negatives)

positive_feature_vectors = list(map(lambda text: calc_feature_vector(features, text), positives))
negative_feature_vectors = list(map(lambda text: calc_feature_vector(features, text), negatives))

data = positive_feature_vectors + negative_feature_vectors
target = np.array(([1] * 1000) + ([0] * 1000))

# pos_files = list(map(lambda file: os.path.join("imdb1.train/pos", file), os.listdir("imdb1.train/pos")))

# test_index = 660
# feature_vector = calc_feature_vector(features, positives[test_index])
# pretty_feature_vector = prettify_feature_vector(feature_vector, features)
# keywords = contained_keywords(feature_vector, features)

# print(pos_files[test_index])
# print(positives[test_index])
# for keyword in keywords:
#     print(keyword)

######################################################################
################### Learning #########################################
######################################################################

# clf = svm.SVC().fit(data, target)
clf = MultinomialNB().fit(data, target)
# clf = tree.DecisionTreeClassifier()
# clf = neighbors.KNeighborsClassifier()

scores = cross_validation.cross_val_score(clf, data, target, cv=10)
print("Accuracy: %.2f (+- %.2f)" % (scores.mean(), (scores.std() * 2)))


# best 10 features 1: ['bad', 'best', 'can', 'emotion', 'enjoy', 'great', 'love', 'money', 'thumbs', 'wors']
# best 10 features 2: ['applauded', 'bad', 'best', 'emotion', 'enjoy', 'great', 'love', 'money', 'thumbs', 'wors']
# best 10 features for SVM: ['awful', 'bad', 'highly', 'money', 'not', 'nothing', 'tear', 'wors']
# best 50 features: ['avoid', 'bad', 'beautiful', 'best', 'better', 'brillian', 'cried', 'different', 'forgotten', 'genius', 'great', 'hate', 'horribl', 'insulting', 'intense', 'juicy', 'laugh', 'like', 'love', 'money', 'must', "n't", 'never', 'off', 'over', 'perfect', 'plot', 'pointless', 'positive', 'problem', 'recommend', 'ruin', 'serious', 'should', 'strong', 'stunning', 'stupid', 'tear', 'tedious', 'tense', 'thank', 'unsatisfied', 'very', 'waste', 'well', 'wonderful', 'wors', 'yet']


# while True:
#     bitset = input()
#     accuracy = calc_accuracy(bitset)
#     print(accuracy)

