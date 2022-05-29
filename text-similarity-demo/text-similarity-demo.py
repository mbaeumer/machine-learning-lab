import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_text_similarity():
    my_text = '''TfidfVectorizer converts a collection of raw documents to a matrix of Tfidf features. Text analysis is a major application field in machine learning algorithms.The text must first to numerical feature vectors because the computer wants to process numeric data.
    Sklearn accomplishes natural language text processing by: tokenizing the strings and assigning an integer for each possible token; counting the occurrences of each token in each document; and normalising and weighting with diminishing importance the tokens that occur in the majority of text documents. When transforming the text data into numeric data, each individual token occurrence frequency is treated as a feature.
    A corpus of documents can be represented as one row of data for each document and one column, or feature, for each token.
    Vectorization is the general process of turning a collection of text documents into numerical feature vectors. This specific strategy is called Bag of Words representation. Documents are described by word occurrences while ignoring the relative position of the words in the document'''

    sk_text = '''Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
    In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
    tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
    counting the occurrences of tokens in each document.
    normalizing and weighting with diminishing importance tokens that occur in the majority of samples / documents.
    In this scheme, features and samples are defined as follows:
    each individual token occurrence frequency (normalized or not) is treated as a feature.
    the vector of all the token frequencies for a given document is considered a multivariate sample.
    A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus.
    We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document'''

    corpus = [my_text, sk_text]

    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['my_doc', 'sk_doc'])

    print(df)
    print(cosine_similarity(trsfm[0:1], trsfm))

# based on: https://medium.com/geekculture/an-easy-way-to-determine-similarity-between-two-strings-of-text-using-python-de9b1b52f022
if __name__ == '__main__':
    calculate_text_similarity()
