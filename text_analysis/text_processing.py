import logging
import operator
import pickle
import re
from itertools import product

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import progressbar
from many_stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from stop_words import get_stop_words
import string

logger = logging.getLogger(__name__)


def lowercase(d):
    return d.lower()


def apostrophes(s):
    return s.replace("'", "")


def letter_digit_underscore(d):
    pat = r'[^\wáéíóúàèìòùÁÉÍÓÚÀÈÌÒÙ]'
    return re.sub(pat, ' ', d)


def remove_space_newline_tab(d):
    return re.sub(r'\s', ' ', d)


def remove_url(d):
    return re.sub(r'http\S+', '', d)


table = str.maketrans(dict.fromkeys(string.punctuation))


def remove_punctuation(d):
    return d.translate(table)


def remove_digits(d):
    return re.sub('\d+', '', d)


DEFAULT_PIPELINE = [
    remove_url, lowercase, apostrophes, letter_digit_underscore,
    remove_space_newline_tab, remove_punctuation, remove_digits
]


def min_length(d, min_l=4):
    return len(d) >= min_l


DEFAULT_VALIDATION = [min_length]


def validate_word(validation, word):
    valid = True
    for validation_fun in validation:
        if not validation_fun(word):
            valid = False
            break
    return valid


def clean_sentence(sentence,
                   pipeline=DEFAULT_PIPELINE,
                   stop_words=[],
                   validation=DEFAULT_VALIDATION,
                   join=False,
                   tokenizer=word_tokenize):
    for process_fun in pipeline:
        if len(sentence) == 0:
            break
        sentence = process_fun(sentence)
    if len(sentence) == 0:
        return ""

    final_sentence = []
    for word in tokenizer(sentence):
        if word in stop_words:
            continue
        if not validate_word(validation, word):
            continue
        final_sentence.append(word)
    if join:
        return ' '.join(final_sentence)
    else:
        return final_sentence


def vocabulary_size(data):
    logging.info('Computing Vocabulary size')
    d = set()
    for sentence in data:
        d.update(w for w in word_tokenize(sentence))
    return len(d)


def clean_content(data,
                  pipeline=DEFAULT_PIPELINE,
                  stop_words=[],
                  validation=DEFAULT_VALIDATION,
                  join=False,
                  tokenizer=word_tokenize):
    """Clean text data

    Arguments:
        data: list of str
              The list of documents to be cleaned

    Keyword Arguments:
        pipeline: list of functions
                   A sequence of function to be applied to every document
                   default=DEFAULT_PIPELINE
        stop_words: iterable
                    Stop words to remove from each document
        join: bool
              Wheter to join the tokenized document or not

    Returns:
        list of str
        Document processed.
    """
    logger.info('Cleaning content')
    documents = []
    for d in progressbar.progressbar(data):
        documents.append(
            clean_sentence(d, pipeline, stop_words, validation, join,
                           tokenizer))
    return documents


def get_stopwords(lang):
    lang_mapping = {
        'en': 'english',
        'es': 'spanish',
        'pt': 'portuguese',
        'it': 'italian'
    }
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
