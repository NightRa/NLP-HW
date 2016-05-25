from lxml import etree
from lxml.builder import E
from random import shuffle
import codecs
import sys
import os
import nltk
import math

class Example:
    # id: String
    # sense: String
    # sentences: [String]

    def __init__(self, id, sense, sentences):
        self.id = id
        self.sense = sense
        self.sentences = sentences

    def text(self):
        return " ".join(self.sentences)

    def tokens(self):
        return nltk.word_tokenize(self.text())

    def to_xml(self):
        context = E.context(*map(E.s, self.sentences))
        return E.instance(E.answer(instance=self.id, senseid=self.sense), context, id=self.id)

    @staticmethod
    def from_xml(node):
        answer = node[0]
        context = node[1]
        assert node.tag == 'instance'
        assert answer.tag == 'answer'
        assert context.tag == 'context'

        id = node.get('id')
        sense = answer.get('senseid')
        sentences = list(map(lambda s: s.text, context))

        return Example(id, sense, sentences)

    def __str__(self):
        return '\n' + self.id + '\n\t' + '\n\t'.join(self.sentences)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


class Corpus:
    # examples: [Example]
    # item: String (line-n)
    # lang: String (en)

    def __init__(self, examples, lang, item):
        self.examples = examples
        self.lang = lang
        self.item = item

    def examples_by_type(self):
        return group_by(lambda example: example.sense, self.examples)

    def token_occurrences_by_sense(self):
        return dict_map_values(lambda examples: bag(flatMap(lambda example: example.tokens(), examples)),
                               self.examples_by_type())

    def to_xml(self):
        examples = map(Example.to_xml, self.examples)
        return E.corpus(E.lexelt(*examples, item=self.item), lang=self.lang)

    @staticmethod
    def from_xml(node):
        lexelt = node[0]
        assert node.tag == 'corpus'
        assert lexelt.tag == 'lexelt'

        lang = node.get('lang')
        item = lexelt.get('item')
        examples = list(map(Example.from_xml, lexelt))
        return Corpus(examples, lang, item)

    @staticmethod
    def from_file(filename):
        root = etree.parse(filename).getroot()
        return Corpus.from_xml(root)

    def pretty_print(self):
        xml = self.to_xml()
        out = etree.tostring(xml, pretty_print=True, encoding='unicode')
        return out

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


##############################################################################
####################### Helper Functions #####################################
##############################################################################

# group all elements sharing the same key, derived by f.
# {k -> {e | f(e)=k}}
def group_by(f, l):
    out = {}
    for e in l:
        key = f(e)
        if not key in out:
            out[key] = [e]
        else:
            out[key].append(e)
    return out

# Count the number of occurrences of each element, return as a dictionary {element -> #occurrences}
def bag(l):
    return dict_map_values(lambda occurrences: len(occurrences), group_by(identity, l))

# How many elements in @l satisfy @p?
def count(l, p):
    count = 0
    for e in l:
        if p(e):
            count += 1
    return count

def identity(x):
    return x

# flatMap: [a], (a -> [b]) -> [b]
def flatMap(f, list):
    list_out = []
    for e in list:
        list_out.extend(f(e))
    return list_out

# [[A]] -> [A]
def flatten(l):
    return [item for sublist in l for item in sublist]

def dict_map_values(f, dict):
    return {k:f(v) for k,v in dict.items()}

# read_file: file path -> IO String
def write_file(file_path, data):
    with codecs.open(file_path, 'w', 'utf-8') as f:
        return f.write(data)



##############################################################################
######################### Naive Bayes ########################################
##############################################################################

def prior(sense, corpus, examples_by_type):
    csk = len(examples_by_type[sense])
    cw = len(corpus.examples)
    return csk / cw

# contexts_tokens_by_sense: Map[Sense -> Map[Word -> #Occurences]]
def posterior(word, sense, token_occurrences_by_sense, sense_num_tokens):
    cvsk = 0 if word not in token_occurrences_by_sense[sense] else token_occurrences_by_sense[sense][word]
    csk_total_words = sense_num_tokens[sense]

    # Do Add-one smoothing
    vocabulary_size = len(token_occurrences_by_sense[sense])
    return (cvsk + 1) / (csk_total_words + vocabulary_size)

def score(sense, example, train_corpus, examples_by_type, token_occurrences_by_sense, sense_num_tokens):
    tokens = example.tokens()
    return math.log(prior(sense, train_corpus, examples_by_type)) + \
           sum(map(lambda word: math.log(posterior(word, sense, token_occurrences_by_sense, sense_num_tokens)), tokens))

def classify(example, train_corpus, examples_by_type, token_occurrences_by_sense, sense_num_tokens):
    senses = token_occurrences_by_sense.keys()
    return max(senses, key=lambda sense: score(sense, example, train_corpus, examples_by_type, token_occurrences_by_sense, sense_num_tokens))

def classify_test_corpus(train_corpus, test_corpus):
    examples_by_type = train_corpus.examples_by_type()
    token_occurrences_by_sense = train_corpus.token_occurrences_by_sense()
    senses = token_occurrences_by_sense.keys()
    sense_num_tokens = dict(map(lambda sense: (sense, sum(token_occurrences_by_sense[sense].values())), senses))
    # [Example -> Sense]
    classifications = list(map(lambda example: (example, classify(example, train_corpus, examples_by_type, token_occurrences_by_sense, sense_num_tokens)),
                               test_corpus.examples))
    return classifications

##############################################################################
######################### Evaluation #########################################
##############################################################################

# (1,1)
def true_positives(classifications, sense):
    def is_true_positive(example, chosen_sense):
        return example.sense == sense and chosen_sense == sense

    return count(classifications, lambda classification: is_true_positive(*classification))

# (0,1)
def false_positives(classifications, sense):
    def is_false_positive(example, chosen_sense):
        return example.sense != sense and chosen_sense == sense

    return count(classifications, lambda classification: is_false_positive(*classification))

# (1,0)
def false_negatives(classifications, sense):
    def is_false_negative(example, chosen_sense):
        return example.sense == sense and chosen_sense != sense

    return count(classifications, lambda classification: is_false_negative(*classification))

def calc_precision(classifications, sense):
    tp = true_positives(classifications, sense)
    fp = false_positives(classifications, sense)
    return tp / (tp + fp)

def calc_recall(classifications, sense):
    tp = true_positives(classifications, sense)
    fn = false_negatives(classifications, sense)
    return tp / (tp + fn)

def calc_total_accuracy(classifications):
    def is_true_positive(example, chosen_sense):
        return example.sense == chosen_sense

    correct = count(classifications, lambda classification: is_true_positive(*classification))
    total = len(classifications)
    return correct / total

##############################################################################
################################ Main ########################################
##############################################################################

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    train_corpus = Corpus.from_file(train_file)
    test_corpus = Corpus.from_file(test_file)

    classifications = classify_test_corpus(train_corpus, test_corpus)
    senses = train_corpus.examples_by_type().keys()

    for sense in sorted(senses):
        precision = calc_precision(classifications, sense)
        recall = calc_recall(classifications, sense)
        print("%s: precision: %.3f, recall %.3f" % (sense, precision, recall))

    total_accuracy = calc_total_accuracy(classifications)
    print("total accuracy: %.3f" % total_accuracy)

    classifications_out_file = codecs.open(output_file, 'w', 'utf-8')
    for example, chosen_sense in classifications:
        classifications_out_file.write(example.id + " " + chosen_sense + "\n")

    classifications_out_file.close()

main()
