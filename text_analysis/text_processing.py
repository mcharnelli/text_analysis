import operator
import pickle
import re
from itertools import product

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from many_stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from stop_words import get_stop_words


def lowercase(d):
    return d.lower()


def apostrophes(s):
    return s.replace("'", "")


def letter_digit_underscore(d):
    pat = r'^\W+|\W+$'
    return re.sub(pat, ' ', d)


def remove_space_newline_tab(d):
    return re.sub(r'\s', ' ', d)


def remove_punctuation(d):
    return re.sub(r'[^a-zA-Z0-9]', ' ', d)


def remove_digits(d):
    return re.sub('\d+', '', d)


DEFAULT_PIPELINE = [
    lowercase, apostrophes, letter_digit_underscore, remove_space_newline_tab,
    remove_punctuation, remove_digits
]


def clean_content(data, pipeline=None, stop_words=[], join=False):
    if pipeline is None:
        pipeline = DEFAULT_PIPELINE
    documents = []
    for d in data:
        w = d
        for process_fun in pipeline:
            w = process_fun(w)
        sentence = [
            word for word in word_tokenize(w) if word not in stop_words
        ]
        if join:
            sentence = ' '.join(sentence)
        documents.append(sentence)
    return documents


def get_stopwords(lang):
    lang_mapping = {'en': 'english', 'es': 'spanish', 'pt': 'portuguese'}
    stop_words = set(get_stop_words(lang))
    nltk_words = set(stopwords.words(lang_mapping[lang]))
    m_stop_words = set(get_stop_words(lang))
    stop_words.update(nltk_words)
    stop_words.update(m_stop_words)
    return stop_words


def get_topics_distribution(categories, min_occurences=12):
    """
    Get Topics Distribution

    Parameters
    ----------
    categories: dict
                A dictionary where the keys are the topics, and the values are the words associated with each topic.
    """
    probabilities = dict()

    for category in categories.keys():
        probabilities[category] = dict()
        for w in categories[category]:
            if w not in probabilities[category]:
                probabilities[category][w] = 0
            probabilities[category][w] = probabilities[category][w] + 1
        remove = []
        for w in probabilities[category].keys():
            if probabilities[category][w] < min_occurences:
                remove.append(w)
        for k in remove:
            del probabilities[category][k]
        cant_words = sum(probabilities[category].values())
        probabilities[category] = {
            k: v / cant_words
            for k, v in probabilities[category].items()
        }

    return probabilities


def get_p_b_d(document, wa, wb):
    """ Given a biterm obtain its probability in the docuemnt """
    n_biterms = 0.0
    biterm = 0.0
    for w1, w2 in product(document, document):
        n_biterms = n_biterms + 1
        if (w1 == wa) and (w2 == wb):
            biterm = biterm + 1
    return biterm / n_biterms


def topic_assignment(question, topic_distribution):
    total_words = sum([len(topic) for topic in topic_distribution.values()])
    p_z = {
        topic_name: len(topic) / total_words
        for topic_name, topic in topic_distribution.items()
    }
    p_z_d = dict()
    for j in topic_distribution.keys():
        p_z_d[j] = 0.0
    for w1, w2 in product(question, question):
        p_b_d = get_p_b_d(question, w1, w2)
        temp = dict()
        for j in topic_distribution.keys():
            t_d_w1 = 0.00000001
            t_d_w2 = 0.00000001
            if w1 in topic_distribution[
                    j] and topic_distribution[j][w1] > 0.00000001:
                t_d_w1 = topic_distribution[j][w1]
            if w2 in topic_distribution[
                    j] and topic_distribution[j][w2] > 0.00000001:
                t_d_w2 = topic_distribution[j][w2]
            temp[j] = p_z[j] * t_d_w1 * t_d_w2
        deno = sum(temp.values()) + 0.00000001
        p_z_b = {j: t / deno for j, t in temp.items()}
        for j in topic_distribution.keys():
            p_z_d[j] = p_z_d[j] + p_z_b[j] * p_b_d
    if all(value <= 0.00000001 * 0.00000001 for value in p_z_d.values()):
        prob = 'general'
    else:
        prob = max(p_z_d.items(), key=operator.itemgetter(1))[0]
    return prob
