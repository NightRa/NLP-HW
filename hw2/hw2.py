import re
import codecs
import math

file = 'hw2/datasets/devset/wikipedia-tokenized.txt'

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

def map_filter_keys(dict, p):
    return {k:v for k,v in dict.items() if p(k)}

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

# raw_frequency: [String] -> Map[(String, String), Double]
def raw_frequency(sentences):
    tokens = get_unigrams(sentences)
    num_tokens = len(tokens)
    bigrams = get_bigrams(sentences)
    return count_probabilities(bigrams, num_tokens)

def bigram_pmi(sentences):
    tokens = get_unigrams(sentences)
    unigram_frequencies = count_probabilities(tokens, len(tokens))
    bigrams = get_bigrams(sentences)
    bigram_frequencies = count_probabilities(bigrams, len(bigrams))
    def pmi(pxy, px, py):
        return math.log(pxy / (px * py), 2)

    return {(x, y): pmi(pxy, unigram_frequencies[x], unigram_frequencies[y])
                    for (x, y), pxy in bigram_frequencies.items()}


def bigram_pmi_filtered(sentences):
    bigram_pmis = bigram_pmi(sentences)
    bigrams = get_bigrams(sentences)
    bigram_freqs = count_frequencies(bigrams)
    return map_filter_keys(bigram_pmis, lambda bigram: bigram_freqs[bigram] > 20)

# trigram_pmi: [String], (pmi_f: (unigrams_f, bigrams_f, trigrams_f, trigram) -> double)) -> double
def trigram_pmi(sentences, pmi_f):
    unigrams = get_unigrams(sentences)
    unigrams_p = count_probabilities(unigrams, len(unigrams))
    bigrams = get_bigrams(sentences)
    bigrams_p = count_probabilities(bigrams, len(bigrams))
    trigrams = get_trigrams(sentences)
    trigrams_p = count_probabilities(trigrams, len(trigrams))
    trigrams_f = count_frequencies(trigrams)
    trigram_pmis = {trigram: math.log(pmi_f(unigrams_p, bigrams_p, trigrams_p, trigram), 2)
                        for trigram in trigrams}
    return map_filter_keys(trigram_pmis, lambda trigram: trigrams_f[trigram] > 20)

def trigram_pmi_a(sentences):
    def pmi_a(unigrams_p, bigrams_p, trigrams_p, trigram):
        x, y, z = trigram
        return trigrams_p[trigram] / (unigrams_p[x] * unigrams_p[y] * unigrams_p[z])
    return trigram_pmi(sentences, pmi_a)

def trigram_pmi_b(sentences):
    def pmi_b(unigrams_p, bigrams_p, trigrams_p, trigram):
        x, y, z = trigram
        return trigrams_p[trigram] / (bigrams_p[(x, y)] * bigrams_p[(y, z)])
    return trigram_pmi(sentences, pmi_b)

def trigram_pmi_c(sentences):
    def pmi_c(unigrams_p, bigrams_p, trigrams_p, trigram):
        x, y, z = trigram
        return trigrams_p[trigram] / (unigrams_p[x] * unigrams_p[y] * bigrams_p[(x, y)] * bigrams_p[(y, z)])
    return trigram_pmi(sentences, pmi_c)

# top: k: Int, Map[A, B Ordered] -> List[(A, B)] Sorted on B, with only top k elements.
def top(k, scored):
    return sorted(scored.items(), key=(lambda pair: pair[1]), reverse=True)[:k]
                                                    # ^ on value

# bigram_pmis: Map[(String, String), Double]
bigram_pmis = bigram_pmi(sentences)

# sorted_frequencies: List[((String, String), Double)]
sorted_frequencies = top(100, bigram_pmis)

for (bigram, pmi) in sorted_frequencies:
  w1, w2 = bigram
  print(w1, w2, '=', "%.3f" % (pmi * 1000))


