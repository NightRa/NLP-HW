from lxml import etree
from lxml.builder import E
from random import shuffle
import codecs
import sys
import os

class Example:
    # id: String
    # sense: String
    # sentences: [String]

    def __init__(self, id, sense, sentences):
        self.id = id
        self.sense = sense
        self.sentences = sentences

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

# split a list to 2 lists, one of size min(n, k) and the other n - min(n,k)
def split(l, k):
    k = min(len(l), k)
    return l[:k], l[k:]

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
#################### Train-Test Sets Split ###################################
##############################################################################

# returns (train_corpus, test_corpus)
def split_train_test(corpus, test_size):
    examples_by_type = corpus.examples_by_type()
    for sense, examples in examples_by_type.items():
        shuffle(examples)

    # {sense -> (test=[Example],train=[Example]) }
    split_examples_by_type = dict_map_values(lambda examples: split(examples, test_size),
                                             examples_by_type)

    test  = flatten(dict_map_values(lambda x: x[0], split_examples_by_type).values())
    train = flatten(dict_map_values(lambda x: x[1], split_examples_by_type).values())

    return Corpus(train, corpus.lang, corpus.item), Corpus(test, corpus.lang, corpus.item)

def task_1(input_file_corpus, output_file_train, output_file_test, test_size):
    corpus = Corpus.from_file(input_file_corpus)
    train, test = split_train_test(corpus, test_size)
    write_file(output_file_train, train.pretty_print())
    write_file(output_file_test, test.pretty_print())

##############################################################################
################################ Main ########################################
##############################################################################

input_file = sys.argv[1]
output_folder = sys.argv[2]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

task_1(input_file,
       os.path.join(output_folder, 'train.xml'),
       os.path.join(output_folder, 'test.xml'),
       test_size=50)
