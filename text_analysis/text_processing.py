from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer,  word_tokenize
from nltk.corpus import stopwords
import operator
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
from many_stop_words import get_stop_words



def clean_content(df, stop_words):
    data = df.astype(str)
    # Converting text to lowercase characters
    data = data.str.lower()
    # Remove apostrophes
    data = data.str.replace("'", ' ')
    # Removing any character which does not match to letter,digit or underscore
    pat = r'^\W+|\W+$'
    data = data.str.replace(pat, ' ')
    # Removing space,newline,tab
    pat = r'\s'
    data = data.str.replace(pat, ' ')
    # Removing punctuation
    pat = r'[^a-zA-Z0-9]'
    data = data.str.replace(pat, ' ')
    # Removing digits
    data = data.str.replace(r'\d+', '')
    # Tokenizing data
    # data.dropna(inplace=True)
    data = data.apply(word_tokenize)
    # Removing stopwords
    data = data.apply(lambda x: [i for i in x if i not in stop_words])

    return data


def cleaning_tokenize(text, ln):
    tokens = get_tokens_text(text)
    tokens = cleaning_stopwords(tokens, ln)
    return tokens


def cleaning_stopwords(vector, ln):
    filtered_vector = []
    for word in vector:
        if word not in get_stopwords(ln):
            filtered_vector.append(word)
    return filtered_vector


def get_tokens_text(text):
    content = text.lower()
    content = re.sub(r'[0-9]+', '', content)
    content = re.sub(r'[^\w]', ' ', content)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(content)
    return tokens


def get_stopwords(lang):
    stop_words = set(get_stop_words(lang))  # About 900 stopwords
    nltk_words = {}
    if lang == 'en':
        nltk_words = set(stopwords.words('english'))  # About 150 stopwords
    if lang == 'es':
        nltk_words = set(stopwords.words('spanish'))  # About 150 stopwords
    if lang == 'pt':
        nltk_words = set(stopwords.words('portuguese'))  # About 150 stopwords
    m_stop_words = set (get_stop_words(lang))
    stop_words.update(nltk_words)
    stop_words.update(m_stop_words)
    return stop_words


def add_other_stopwords(stop_words, other_stop_words):
    stop_words.update(other_stop_words)
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
        probabilities[category] = {k: v / cant_words for k, v in probabilities[category].items()}

    return probabilities


def get_p_b_d(document, wa, wb):
    """ Given a biterm obtain its probability in the docuemnt """
    n_biterms = 0.0
    biterm = 0.0
    for w1, w2 in product(document, document):
        n_biterms = n_biterms + 1
        if (w1 == wa) and (w2 == wb):
            biterm = biterm + 1
    return biterm/n_biterms


def topic_assignment(question, topic_distribution):
    total_words = sum([len(topic) for topic in topic_distribution.values()])
    p_z = {topic_name: len(topic) / total_words
           for topic_name, topic in topic_distribution.items()}
    p_z_d = dict()
    for j in topic_distribution.keys():
        p_z_d[j] = 0.0
    for w1, w2 in product(question, question):
        p_b_d = get_p_b_d(question, w1, w2)
        temp = dict()
        for j in topic_distribution.keys():
            t_d_w1 = 0.00000001
            t_d_w2 = 0.00000001
            if w1 in topic_distribution[j] and topic_distribution[j][w1] > 0.00000001:
                    t_d_w1 = topic_distribution[j][w1]
            if w2 in topic_distribution[j] and topic_distribution[j][w2] > 0.00000001:
                    t_d_w2 = topic_distribution[j][w2]
            temp[j] = p_z[j] * t_d_w1 * t_d_w2
        deno = sum(temp.values()) + 0.00000001
        p_z_b = {j: t / deno for j,t in temp.items()}
        for j in topic_distribution.keys():
            p_z_d[j] = p_z_d[j] + p_z_b[j] * p_b_d
    if all(value <= 0.00000001*0.00000001 for value in p_z_d.values()):
        prob = 'general'
    else:
        prob = max(p_z_d.items(), key=operator.itemgetter(1))[0]
    return prob


if __name__ == '__main__':
    topics_dist = get_topics_distribution()
    #max_word = sorted(topics_dist['basketball'].items(), key=operator.itemgetter(1))[::-1][1:10]
    #print(max_word)
    print("prob de kung en artes marciales " + str(topics_dist['martial arts']['kung']))
    print (topics_dist['tennis'])
    #question = 'How many points does a volleyball set have?'
    # me la clasifica de beach volleyball y no volleyball
    question = 'Where is Kung Fu from? USA Japan Antarctica China USA Japan Antarctica China'
    question = 'What age was Maria Sharapova when she won Wimbledon in 2004? 21 14 17 18 21 14 17 18'
    vector = cleaning_tokenize(question)
    print(vector)
    category = get_category(vector)
    print(category)
