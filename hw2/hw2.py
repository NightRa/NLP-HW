import re

import codecs

file = 'datasets/devset/childes-tokenized.txt'

with codecs.open(file, 'r', 'utf-8') as f:
    text = f.read()

sentences = re.sub(r"(\r\n)+|\n+|\xa0|\t", "\n", text).split("\n")

# flatMap: [a], (a -> [b]) -> [b]
def flatMap(list, f):
    list_out = []
    for e in list:
        list_out.extend(f(e))
    return list_out

# returns the sliding windows of size n of the list.
# window: [A] -> [(a1, a2, ..., an)]
def window(list, n):
    return [tuple(list[i:i + n]) for i in range(len(list) - n + 1)]

# n_grams: (lines: [String]) -> [(w1, w2, ..., wn)]
def n_grams(sentences, n):
    def sentence_grams(sentence):
        tokens = sentence.split(' ')
        return window(tokens, n)
    return flatMap(sentences, sentence_grams)

# get_unigrams: [Sentence] -> [Token = String]
def get_unigrams(sentences):
    # unpack the tuple of length 1
    return [t for (t,) in n_grams(sentences, 1)]

# get_bigrams: [String] -> [(String, String)]
def get_bigrams(sentences):
    return n_grams(sentences, 2)

# get_trigrams: [String] -> [(String, String, String)]
def get_trigrams(sentences):
    return n_grams(sentences, 3)

# count_frequencies: List[A] -> Map[A, Int (#occurences)]
def count_frequencies(elements):
    # foldMap (Map(_ -> 1))
    counts = {}
    for e in elements:
        if e not in counts:
            counts[e] = 1
        else:
            counts[e] += 1
    return counts

# problem: we were asked to divide by #unigrams in raw_frequency, not by #bigrams.
# count_probabilities: List[A] -> Map[A, Double (prob. max likelihood estimation)]
def count_probabilities(elements, total):
    frequencies = count_frequencies(elements)
    def normalize(freq):
        return freq / total
    # frequencies.map(normalize)
    return {k: normalize(v) for k, v in frequencies.items()}


def raw_frequency(sentences):
    tokens = get_unigrams(sentences)
    num_tokens = len(tokens)
    bigrams = get_bigrams(sentences)
    return count_probabilities(bigrams, num_tokens)

# frequencies: Map[(String, String), Double]
frequencies = raw_frequency(sentences)

# sorted_frequencies: List[((String, String), Double)]
sorted_frequencies = sorted(frequencies.items(), key=(lambda pair: pair[1]), reverse=True)
                                                        # ^ freq
for (bigram, freq) in sorted_frequencies:
  w1, w2 = bigram
  print(w1, w2, '=', "%.3f" % (freq * 1000))
