import gzip
import logging
import pickle
import re
import time

import matplotlib.pyplot as plt
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import textmining.BTM as BTM
import textmining.text_processing as ca
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud


def plot_words_frecuencies(most_common):
    fig, ax = plt.subplots(1, 1)
    ax.barh(range(len(most_common)), [val[1] for val in most_common], align='center')
    ax.yticks(range(len(most_common)), [val[0] for val in most_common])
    return fig

#get n-words frecuencies
def calculate_words_frecuencies(lists_words, n=30):
    words = ca.cleaning_tokenize(lists_words)
    word_freqdist = nltk.FreqDist(words)
    most_common = word_freqdist.most_common(n)
    return most_common

#get bigram frecuencies
def calculate_bigram_frecuencies(lists_words):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    words = ca.cleaning_tokenize(lists_words)
    finder = BigramCollocationFinder.from_words(words)
    finder.nbest(bigram_measures.pmi, 5)
    return finder


def add_other_stopwords(stop_words, other_stop_words):
    stop_words.update(other_stop_words)
    return stop_words


def get_word_cloud(df, stop_words, file, collocations, color='white'):
    logging.info('Generating word cloud')
    wordcloud = WordCloud(
        background_color=color,
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        collocations=collocations,
        random_state=42
        ).generate(df)
    return wordcloud


def get_best_kmeans_k_parameter(data, filename, clusters, iter):
    vect = TfidfVectorizer()
    x = vect.fit_transform(data.values)
    sse = {}
    for k in range(1, clusters):
        logging.info("Number of topics:" + str(k))
        kmeans = MiniBatchKMeans(n_clusters=k,
                init='k-means++',
                max_iter=iter,
                batch_size=int(x.shape[0]/20)).fit(x)
        sse[k] = kmeans.inertia_  # Inertia: Sum of squared distances of samples to their closest cluster center.
    return sse


def compute_coherence_values(dictionary, corpus, texts, clusters, iter=100):
    coherence_values = []
    model_list = []
    for num_topics in range(2, clusters):
        logging.info("Number of topics:" + str(num_topics))
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=iter)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def get_best_k_parameter_lda(data, filename, clusters, iter):
    # create a Gensim dictionary from the texts
    dictionary = corpora.Dictionary(data)
    # remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    # convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in data]
    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data, clusters=clusters, iter=iter)
    # Show graph
    x = range(2, clusters)
    return x, coherence_values
