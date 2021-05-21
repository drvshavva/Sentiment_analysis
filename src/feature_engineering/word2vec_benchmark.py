from os.path import dirname
from gensim.models.word2vec import Word2Vec
import multiprocessing
from sklearn import utils
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

MODEL_PATH = dirname(dirname(dirname(__file__))) + "/models/word2vec/"


def train_word2vec(corpus, n_epoch, name_corpus, sg, vector_size, negative, window, min_count, alpha, min_alpha):
    cores = multiprocessing.cpu_count()
    model = Word2Vec(sg=sg, size=vector_size, negative=negative, window=window, min_count=min_count, workers=cores,
                     alpha=alpha, min_alpha=min_alpha)
    model.build_vocab([x.words for x in tqdm(corpus)])

    for epoch in range(n_epoch):
        model.train(utils.shuffle([x.words for x in tqdm(corpus)]), total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.save(
        f"{MODEL_PATH}word2vec_{name_corpus}_sg_{sg}_size_{vector_size}_window_{window}_min_count_{min_count}.model")
    return model
