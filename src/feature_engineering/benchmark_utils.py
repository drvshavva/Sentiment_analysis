from sklearn import linear_model, metrics, svm
from sklearn import ensemble
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from collections import Counter


def classification_report(x_train, x_test, y_train, y_test):
    models = [('LogisticRegression', linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial')),
              ('RandomForest', ensemble.RandomForestClassifier(n_estimators=100)),
              ('SVM', svm.SVC())]

    for name, model in models:
        clf = model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(f"{name}:")
        print(f"accuracy: {metrics.accuracy_score(y_pred=y_pred, y_true=y_test)}")
        print(f"precision: {metrics.precision_score(y_pred=y_pred, y_true=y_test, average='macro')}")
        print(f"recall: {metrics.recall_score(y_pred=y_pred, y_true=y_test, average='macro')}")
        print(f"{metrics.classification_report(y_pred=y_pred, y_true=y_test)}")


def get_word_counts(data):
    words = data.tweet.to_string().split()
    return Counter(words)


def labelize_tweets_ug(tweets, label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


def get_mean_vector(model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in model.wv]
    if len(words) >= 1:
        return np.mean(model[words], axis=0)
    else:
        return np.zeros((1, model.vector_size))


def get_vectors(model, corpus):
    vectors = []
    for sentence in corpus:
        vec = get_mean_vector(model, sentence)
        vectors.append(vec)
    return vectors


def get_max_len_sentence(series):
    res = series.str.split().str.len().max()

    print(f"The maximum length in words are : {res}")
