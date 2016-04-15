import re
import codecs
import math
import os
import sys

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

def dict_filter_keys(dict, p):
    return {k:v for k,v in dict.items() if p(k)}

def dict_map_values(dict, f):
    return {k:f(v) for k,v in dict.items()}

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

# bigram_pmi_filtered: k:Int, [String] -> Map[Bigram, PMI], s.t. each bigram appears at least k times in the corpus.
def bigram_pmi_filtered(sentences, k):
    bigram_pmis = bigram_pmi(sentences)
    bigrams = get_bigrams(sentences)
    bigram_freqs = count_frequencies(bigrams)
    return dict_filter_keys(bigram_pmis, lambda bigram: bigram_freqs[bigram] > k)

# trigram_pmi: [String], (pmi_f: (unigrams_p, bigrams_p, trigrams_p, trigram) -> double)), k: Int -> Map[Trigram, PMI],
# s.t. each trigram appears at least @k times in the corpus.
def trigram_pmi(sentences, pmi_f, k):
    unigrams = get_unigrams(sentences)
    unigrams_p = count_probabilities(unigrams, len(unigrams))
    bigrams = get_bigrams(sentences)
    bigrams_p = count_probabilities(bigrams, len(bigrams))
    trigrams = get_trigrams(sentences)
    trigrams_p = count_probabilities(trigrams, len(trigrams))
    trigrams_f = count_frequencies(trigrams)
    trigram_pmis = {trigram: math.log(pmi_f(unigrams_p, bigrams_p, trigrams_p, trigram), 2)
                        for trigram in trigrams}
    return dict_filter_keys(trigram_pmis, lambda trigram: trigrams_f[trigram] > k)

def pmi_a(unigrams_p, bigrams_p, trigrams_p, trigram):
    x, y, z = trigram
    return trigrams_p[trigram] / (unigrams_p[x] * unigrams_p[y] * unigrams_p[z])

def pmi_b(unigrams_p, bigrams_p, trigrams_p, trigram):
    x, y, z = trigram
    return trigrams_p[trigram] / (bigrams_p[(x, y)] * bigrams_p[(y, z)])

def pmi_c(unigrams_p, bigrams_p, trigrams_p, trigram):
    x, y, z = trigram
    return trigrams_p[trigram] / (unigrams_p[x] * unigrams_p[y] * bigrams_p[(x, y)] * bigrams_p[(y, z)])

# top: k: Int, Map[A, B Ordered] -> List[(A, B)] Sorted on B, with only top k elements.
def top(k, scored):
    return sorted(sorted(scored.items(), key=lambda pair: pair[0]), key = lambda pair: pair[1], reverse=True)[:k]
# Sort on score (index 1), then by the (bi/tri)gram lexicographically. (index 0)
# No proper handling of Orders in python, so we can't use the lexicographical ordering imposed by tuples.
#   (we need to partially reverse the order)


###############################################################################################
############################# Formatting ######################################################
###############################################################################################

# input: List[bigram/trigram], output: longest bigram/trigram when shown with spaces.
def longest_collocation(collocations):
    lengths = map(lambda collocation: len(" ".join(collocation)), collocations)
    return max(lengths)

def pad(str, size):
    return str + " " * (size - len(str))

# input: Map[Collocation, metric (raw freq, pmi)], output: Tabular string view.
def format_collocations_metric(collocations, top_size = 100, spaces_between_collocation_and_score = 2):
    # sorted_frequencies: List[(Collocation, Double)]
    sorted_frequencies = top(top_size, collocations)
    max_length = longest_collocation(map(lambda pair: pair[0], sorted_frequencies))
                                                    # ^ The collocation itself from the top list.
    def format_collocation(collocation, score):
        return pad(" ".join(collocation), max_length + spaces_between_collocation_and_score) + "%.3f" % score
        #                                 ^ to have a space between the collocations and the score.
    return "\r\n".join(map(lambda collocation_score: format_collocation(*collocation_score), sorted_frequencies))

# normalized_raw_frequencies: [Sentence: String] -> Map[Bigram, Freq * 1000 : Double]
def formatted_raw_frequencies(sentences):
    frequencies = raw_frequency(sentences)
    return dict_map_values(frequencies, lambda freq: freq * 1000)


###############################################################################################
########################### Input & Output ####################################################
###############################################################################################

def file_sentences(file):
    with codecs.open(file, 'r', 'utf-8') as f:
        text = f.read()

    return re.sub(" +", " ", re.sub(r"(\r\n)+|\n+|\xa0|\t", "\n", text)).split("\n")

def all_texts(input_folder):
    files = map(lambda file: os.path.join(input_folder, file), os.listdir(input_folder))
    return flatMap(files, file_sentences)

def write_to_file(folder, file, body):
    with codecs.open(os.path.join(folder, file), 'w', 'utf-8') as f:
        f.write(body)

def output_all_collocations_metrics(sentences, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    raw_frequencies = formatted_raw_frequencies(sentences)
    write_to_file(output_folder, 'freq_raw.txt', format_collocations_metric(raw_frequencies))
    bigram_pmis = bigram_pmi_filtered(sentences, 20)
    write_to_file(output_folder, 'pmi_pair.txt', format_collocations_metric(bigram_pmis))
    trigram_pmis_a = trigram_pmi(sentences, pmi_a, 20)
    write_to_file(output_folder, 'pmi_tri_a.txt', format_collocations_metric(trigram_pmis_a))
    trigram_pmis_b = trigram_pmi(sentences, pmi_b, 20)
    write_to_file(output_folder, 'pmi_tri_b.txt', format_collocations_metric(trigram_pmis_b))
    trigram_pmis_c = trigram_pmi(sentences, pmi_c, 20)
    write_to_file(output_folder, 'pmi_tri_c.txt', format_collocations_metric(trigram_pmis_c))

input_folder = sys.argv[1]
output_folder = sys.argv[2]

sentences = all_texts(input_folder)
output_all_collocations_metrics(sentences, output_folder)
