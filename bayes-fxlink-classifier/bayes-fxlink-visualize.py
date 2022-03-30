import numpy as np, pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    sns.set()
    df = pd.read_csv("/Users/martinbaumer/Downloads/fxlink-data.csv")
    df.shape
    df.head()
    categories = df.category
    categories.head()
    df.info()

    category_id_df = df[['category']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)
    #id_to_category = dict(category_id_df[['category_id', 'Product']].values)

    import matplotlib.pyplot as plt
    #fig = plt.figure(figsize=(8, 6))
    #df.groupby('category').text.count().plot.bar(ylim=0)
    #plt.show()

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'],
                                                        random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X = count_vect.transform([
            "stackoverflow blog stackoverflow blog text developer ben popper"])
    print(X_train_counts.toarray())
    print(X_train_counts[0,1762])
    print(X_train_counts.shape)
    print(X_train.shape)
    print(y_train.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    print(tfidf_train.shape)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    print(tfidf_test.shape)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    print(clf.predict(count_vect.transform([
            "stackoverflow blog stackoverflow blog text developer ben popper"])))

    print(clf.predict(count_vect.transform([
                                               "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))

    #print("The accuracy is {}".format(accuracy_score(y_test, predicted_categories)))


if __name__ == '__main__':
    main()