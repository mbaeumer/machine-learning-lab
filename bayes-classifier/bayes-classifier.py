import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# based on: https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
def main():
    # Load the dataset
    data = fetch_20newsgroups(categories=None)
    # Get the text categories
    text_categories = data.target_names
    # define the training set
    train_data = fetch_20newsgroups(subset="train", categories=text_categories)
    # define the test set
    test_data = fetch_20newsgroups(subset="test", categories=text_categories)


    print("We have {} unique classes".format(len(text_categories)))
    print("We have {} training samples".format(len(train_data.data)))
    print("We have {} test samples".format(len(test_data.data)))

    # let’s have a look as some training data
    print(test_data.data[5])

    # Build the model
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    # Train the model using the training data
    model.fit(train_data.data, train_data.target)
    # Predict the categories of the test data
    predicted_categories = model.predict(test_data.data)

    print("We have {} unique classes".format(len(text_categories)))
    print("We have {} training samples".format(len(train_data.data)))
    print("We have {} test samples".format(len(test_data.data)))
    print("We have {} predictions".format(len(predicted_categories)))

    print(np.array(test_data.target_names)[predicted_categories])

    print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))


if __name__ == "__main__":
    # based on this article: https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
    main()