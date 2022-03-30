import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    sns.set()
    print("Hello")
    df = pd.read_csv("/Users/martinbaumer/Downloads/fxlink-data-stackoverflog.csv")
    df.shape
    df.head()
    categories = df.category
    categories.head()
    uniq = pd.Series(categories.values).unique()

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=7)

    print("We have {} unique classes".format(len(uniq)))
    print("We have {} training samples".format(len(x_train)))
    print("We have {} test samples".format(len(x_test)))

    #tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    # DataFlair - Fit and transform train set, transform test set
    #tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    #tfidf_test = tfidf_vectorizer.transform(x_test)

    # Build the model
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    # Train the model using the training data
    #model.fit(train_data.data, train_data.target)
    model.fit(x_train, y_train)
    # Predict the categories of the test data
    predicted_categories = model.predict(x_test)

    print("We have {} unique classes".format(len(uniq)))
    print("We have {} training samples".format(len(x_train)))
    print("We have {} test samples".format(len(x_test)))
    print("We have {} predicted categories".format(len(predicted_categories)))


    index = 0
    while index < len(predicted_categories):
        print(x_test.values[index])
        print(predicted_categories[index])
        index = index + 1

    # plot the confusion matrix
    mat = confusion_matrix(y_test, predicted_categories)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=uniq,
                yticklabels=uniq)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()

    print("The accuracy is {}".format(accuracy_score(y_test, predicted_categories)))

    url="stackoverflow blog"
    result = model.predict([url])
    print(result)

    #for row in df.values:
        #print(row[0])
    #df.shape
    #df.head()
    #labels = df.text
    #labels.head()
    #print(labels)

if __name__ == "__main__":
    main()