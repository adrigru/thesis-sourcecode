import pandas as pd
import numpy as np
import argparse
import os
import nltk
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
import logging
import sys
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument("--reduced_classes", type=bool, default=False)

    # Data directories
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", '.'))
    parser.add_argument("--filename", type=str, required=True)

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def get_data(data_dir, filename):
    """
    Get the training data and convert to tensors
    """
    df = pd.read_csv(os.path.join(data_dir, filename), delimiter='\t')

    return df


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, epochs=30)) for doc in sents])
    return np.array(targets), np.array(regressors)


def fit(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression(n_jobs=1, C=1, class_weight='balanced', max_iter=3000, random_state=69420)
    logreg.fit(x_train, y_train)
    preds = logreg.predict(x_test)

    kappa = cohen_kappa_score(preds, y_test, weights='quadratic')
    acc = accuracy_score(y_test, preds)

    # returns the loss and predictions
    return kappa, acc


def run_cv(data_x, labels, k):
    kf = StratifiedKFold(n_splits=k, random_state=69420, shuffle=True)

    history = []

    data_x = data_x.to_numpy()
    labels = labels.to_numpy()

    if reduced_classes:
        print('reduced')
        labels[labels == 4] = 3

    for train_index, test_index in kf.split(data_x, y=labels, groups=None):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        _df_train = pd.DataFrame({'x': X_train, 'y': y_train})
        _df_test = pd.DataFrame({'x': X_test, 'y': y_test})

        train_tagged = _df_train.apply(lambda row: TaggedDocument(words=tokenize_text(row.x), tags=[row.y]), axis=1)
        test_tagged = _df_test.apply(lambda row: TaggedDocument(words=tokenize_text(row.x), tags=[row.y]), axis=1)

        del _df_train, _df_test

        model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=1,
                            min_alpha=0.065,
                            alpha=0.065,
                            seed=69420)

        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=1, seed=69420)

        model_dmm.build_vocab([x for x in train_tagged.values])

        model_dbow.build_vocab([x for x in train_tagged.values])

        model_dmm.train(train_tagged, total_examples=train_tagged.shape[0], epochs=1)
        model_dbow.train(train_tagged, total_examples=train_tagged.shape[0], epochs=1)

        model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

        ys_train, xs_train = get_vectors(model, train_tagged)
        ys_test, xs_test = get_vectors(model, test_tagged)

        kappa, acc = fit(xs_train, xs_test, ys_train, ys_test)
        history.append({'val_qwk': kappa, 'val_acc': acc})

    # For each epoch store mean QWK across folds

    logger.info(
        f"Best average val QWK over {k}-folds: {np.mean([metric['val_qwk'] for metric in history])}")
    logger.info(
        f"Best average accuracy over {k}-folds: {np.mean([metric['val_acc'] for metric in history])}")
    logger.info(f"QWK for each fold: {[metric['val_qwk'] for metric in history]}")
    logger.info(f"Acc for each fold: {[metric['val_acc'] for metric in history]}")

    return history


if __name__ == '__main__':
    args, _ = parse_args()

    cores = multiprocessing.cpu_count()

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'output.log', 'a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info(
        f' epochs={args.epochs},'
        f'reduced_clases={args.reduced_classes},'
        f' k={args.k},'
    )

    np.random.seed(69420)

    _data_dir = args.data_dir
    _k = args.k
    _filename = args.filename
    epochs = args.epochs
    reduced_classes = args.reduced_classes

    df = get_data(_data_dir, _filename)

    df.text_de = df.text_de.apply(str.lower)

    logger.info('DE')

    run_cv(df.text_de, df.label, _k)
