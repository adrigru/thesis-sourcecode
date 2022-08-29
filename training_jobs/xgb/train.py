import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

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
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=4)

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

    labels = df.label.to_numpy()
    embeddings = df.drop(columns=['label']).to_numpy()

    return embeddings, labels

# def get_data(data_dir, filename):
#     """
#     Get the training data and convert to tensors
#     """
#
#     # tensor_dataset = torch.load(os.path.join(data_dir, 'dataset_old.pt'))
#     tensor_dataset = torch.load(os.path.join(data_dir, filename), map_location=torch.device('cpu'))
#
#     return tensor_dataset[:]


def get_class_weights(y_train):
    return compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train),
                                y=y_train.tolist())


def fit(x_train, x_test, y_train, y_test, lr, epochs, depth):
    class_weights = get_class_weights(y_train)
    _cw = np.zeros(y_train.shape)
    _cw[y_train == 0] = class_weights[0]
    _cw[y_train == 1] = class_weights[1]
    _cw[y_train == 2] = class_weights[2]
    _cw[y_train == 3] = class_weights[3]
    if not reduced_classes:
        _cw[y_train == 4] = class_weights[4]

    d_train = xgb.DMatrix(data=x_train, label=y_train, weight=_cw)
    d_test = xgb.DMatrix(data=x_test, label=y_test)

    param = {'max_depth': depth, 'eta': lr, 'objective': 'multi:softmax', 'eval_metric': 'mlogloss', 'num_class': np.unique(y_train).shape[0],
             'seed': 69420}

    evallist = [(d_test, 'eval'), (d_train, 'train')]

    bst = xgb.train(param, d_train, epochs, evallist)

    preds = bst.predict(d_test)
    kappa = cohen_kappa_score(preds, y_test, weights='quadratic')
    acc = accuracy_score(y_test, preds)

    # returns the loss and predictions
    return kappa, acc


def run_cv(data_x, labels, k, lr, epochs, depth):
    kf = StratifiedKFold(n_splits=k, random_state=69420, shuffle=True)

    history = []

    if reduced_classes:
        labels[labels == 4] = 3

    for train_index, test_index in kf.split(data_x, y=labels, groups=None):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        kappa, acc = fit(X_train, X_test, y_train, y_test, lr, epochs, depth)
        history.append({'val_qwk': kappa, 'val_acc': acc})

    # For each epoch store mean QWK across folds

    logger.info(
        f"Best average val QWK over {k}-folds: {np.mean(([metric['val_qwk'] for metric in history]))}")
    logger.info(
        f"Best average accuracy over {k}-folds: {np.mean(([metric['val_acc'] for metric in history]))}")
    logger.info(f"QWK for each fold: {[metric['val_qwk'] for metric in history]}")
    logger.info(f"Acc for each fold: {[metric['val_acc'] for metric in history]}")

    return history


if __name__ == "__main__":
    args, _ = parse_args()

    np.random.seed(69420)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'output.log', 'a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    _learning_rate = args.learning_rate
    _data_dir = args.data_dir
    _k = args.k
    _depth = args.depth
    _filename = args.filename
    epochs = args.epochs
    reduced_classes = args.reduced_classes


    logger.info(
        f' learning_rate={_learning_rate},'
        f' k={_k},'
        f' depth={_depth},'
        f' epochs={epochs},'
        f' reduced_classes={args.reduced_classes},'
        f' filename={args.filename},'
    )

    X, y = get_data(_data_dir, _filename)

    logger.info(f'Starting execution: {datetime.now().isoformat()}')

    history = run_cv(X, y, _k, _learning_rate, epochs, _depth)

    # plot_confusion_matrix(history)
    # print(history)
