from os.path import dirname
from gensim.models import Doc2Vec
import multiprocessing
from sklearn import utils

MODEL_PATH = dirname(dirname(dirname(__file__))) + "/models/doc2vec/"


def train_doc2vec(corpus, n_epoch, name_corpus, vector_size, negative, window, min_count, alpha, min_alpha):
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(size=vector_size, negative=negative, window=window, min_count=min_count, workers=cores, alpha=alpha,
                    min_alpha=min_alpha)
    model.build_vocab(corpus)

    for epoch in range(n_epoch):
        model.train(utils.shuffle(corpus), total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.save(
        f"{MODEL_PATH}Doc2Vec_{name_corpus}_size_{vector_size}_window_{window}_min_count_{min_count}.model")
    return model
