import numpy as np, pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# based on: https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
def main():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    count_vectorizer_demo(corpus)
    tfidf_vectorizer_demo(corpus)



def count_vectorizer_demo(corpus):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(corpus)
    X = count_vect.transform(corpus)
    print(count_vect.get_feature_names())
    print(X_train_counts.toarray())
    print(X_train_counts.shape)
    print(X.shape)
    print(X)
    print(X[0, 1])
    print(X_train_counts[0, 1])

def tfidf_vectorizer_demo(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(corpus)
    print(tfidf_train.toarray())

if __name__ == '__main__':
    main()
