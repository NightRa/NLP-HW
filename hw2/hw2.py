from hw1.nlp2 import *
from hw1.nlp3 import *
import re

import codecs

file = 'datasets/devset/childes-tokenized.txt'

with codecs.open(file, 'r', 'utf-8') as f:
    text = f.read()

sentences = re.sub(r"(\r\n)+|\n+|\xa0|\t", "\n", text).split("\n")

# tokens: (lines: [String]) -> (tokens: [String])
def get_tokens(sentences):
    # sentences.flatMap(_.split(' '))
    tokens_out = []
    for sentence in sentences:
        tokens_out.extend(sentence.split(' '))
    return tokens_out

# pairs :: String -> [(String, String)]
def get_pairs(sentence):
    # sentence.window(2)
    tokens = sentence.split(' ')
    pairs_out = []
    for i in range(0, len(tokens) - 1):
        pairs_out.append((tokens[i], tokens[i + 1]))
    return pairs_out

# pairsFromSentences :: [String] -> [(String, String)]
def get_bigrams(sentences):
    # sentences.flatMap(pairs)
    pairs_out = []
    for sentence in sentences:
        pairs_out.extend(get_pairs(sentence))
    return pairs_out

def get_triples(sentence):
    # sentence.window(3)
    tokens = sentence.split(' ')
    pairs_out = []
    for i in range(0, len(tokens) - 1):
        pairs_out.append((tokens[i], tokens[i + 1]))
    return pairs_out

# trigrams: [String] -> [(String, String, String)]
def get_trigrams(sentences):
    # sentences.flatMap(triples)
    triples_out = []
    for sentence in sentences:
        triples_out.extend(get_triples(sentence))
    return triples_out

# countFrequencies: [(String, String)] -> Map (String, String) -> Int
def count_frequencies(bigrams):
    # foldMap (Map(_ -> 1))
    counts = {}
    for bigram in bigrams:
        if bigram not in counts:
            counts[bigram] = 1
        else:
            counts[bigram] += 1
    return counts

def raw_frequency(sentences):
    tokens = get_tokens(sentences)
    num_tokens = len(tokens)
    bigrams = get_bigrams(sentences)
    frequencies = count_frequencies(bigrams)
    def normalize(freq):
        return (freq / num_tokens) * 1000
    # frequencies.map(normalize)
    return {k: normalize(v) for k, v in frequencies.items()}

# frequencies: Map[(String, String), Double]
frequencies = raw_frequency(sentences)

# sorted_frequencies: List[((String, String), Double)]
sorted_frequencies = sorted(frequencies.items(), key=(lambda pair: pair[1]), reverse=True)
                                                        # ^ freq
for (bigram, freq) in sorted_frequencies:
  w1, w2 = bigram
  print(w1, w2, '=', "%.3f" % freq)
