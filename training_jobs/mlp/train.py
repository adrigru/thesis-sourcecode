import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Classifier(nn.Module):
    def __init__(self, embeddings_size, n_classes):
        super(Classifier, self).__init__()
        self.embeddings_size = embeddings_size
        self.n_classes = n_classes
        self.h1_n = 32

        self.fc1 = nn.Linear(self.embeddings_size, self.h1_n)
        self.fc2 = nn.Linear(self.h1_n, self.n_classes)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(self.h1_n)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    # define the forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


def get_classifier(embeddings_size, n_classes):
    classifier = Classifier(embeddings_size, n_classes)

    return classifier


def plot_metrics(metrics):
    xs = np.arange(0, len(metrics[0]['train_loss']))
    fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True)
    ax1[0].set_title('Train loss')
    ax1[1].set_title('Validation loss')
    ax2[0].set_title('Train QWK')
    ax2[1].set_title('Val QWK')

    ax1[0].plot(xs, np.mean([metric['train_loss'] for metric in metrics], axis=0))
    ax1[1].plot(xs, np.mean([metric['val_loss'] for metric in metrics], axis=0))
    ax2[0].plot(xs, np.mean([metric['train_qwk'] for metric in metrics], axis=0))
    ax2[1].plot(xs, np.mean([metric['val_qwk'] for metric in metrics], axis=0))

    handles, labels = ax1[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.show()


def plot_confusion_matrix(history):
    idcs = [np.argmax(metric['val_qwk']) for metric in history]

    cms = [metric['c_matrix'][i] for metric, i in zip(history, idcs)]

    f, axes = plt.subplots(1, 1, figsize=(20, 8), )

    for i, cm in enumerate(cms):
        disp = ConfusionMatrixDisplay(cm, display_labels=np.arange(0, 5))
        disp.plot(ax=axes, xticks_rotation=45)
        disp.ax_.set_title(f'Fold #: {i}')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    plt.title('Confusion matrix')
    plt.savefig('confusion_matrix_4.png', dpi=150, bbox_inches='tight')


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--t_0", type=int, default=10)
    parser.add_argument("--t_mult", type=int, default=2)
    parser.add_argument("--reduced_classes", type=bool, default=False)

    # Data directories
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", '.'))
    parser.add_argument("--filename", type=str, required=True)

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", '.'))

    return parser.parse_known_args()


def get_data(data_dir, filename):
    """
    Get the training data and convert to tensors
    """
    df = pd.read_csv(os.path.join(data_dir, filename), delimiter='\t')

    ys = torch.tensor(df.label.to_numpy()).long()
    xs = torch.tensor(df.drop(columns=['label']).to_numpy()).float()

    return xs, ys, df


def init_dataloaders(x_train, y_train, x_val, y_val, batch_size=64):
    train_data = TensorDataset(x_train, torch.tensor(y_train.tolist()))
    train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)
    val_data = TensorDataset(x_val, torch.tensor(y_val.tolist()))
    val_dataloader = DataLoader(val_data, sampler=None, batch_size=batch_size)

    return train_dataloader, val_dataloader


def get_class_weights(y_train):
    return compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train),
                                y=y_train.tolist())


def train_epoch(train_dl, model, optimizer, criterion, epoch, scheduler=None):
    total_loss, qwk, acc, f1_micro, f1_macro, f1_weighted = 0, 0, 0, 0, 0, 0

    model.train()
    # https://arxiv.org/pdf/1608.03983.pdf
    n_batches = len(train_dl)
    for i, batch in enumerate(train_dl):
        batch = [r.to(device) for r in batch]
        embeddings, labels = batch
        model.zero_grad()
        preds = model(embeddings)
        loss = criterion(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward(retain_graph=True)
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step(epoch + i / n_batches)
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        labels = labels.detach().cpu().numpy()
        kappa = cohen_kappa_score(preds, labels, weights='quadratic')
        acc += accuracy_score(labels, preds)
        qwk += kappa
        f1_micro += f1_score(labels, preds, average='micro')
        f1_macro += f1_score(labels, preds, average='macro')
        f1_weighted += f1_score(labels, preds, average='weighted')

    # compute the training loss of the epoch
    avg_loss = total_loss / n_batches
    qwk = qwk / n_batches
    acc = acc / n_batches
    f1_micro = f1_micro / n_batches
    f1_macro = f1_macro / n_batches
    f1_weighted = f1_weighted / n_batches

    # returns the loss and predictions
    return avg_loss, qwk, acc, f1_micro, f1_macro, f1_weighted


def evaluate_epoch(val_dl, model, criterion):
    model.eval()

    total_loss, v_qwk, v_acc, f1_micro, f1_macro, f1_weighted = 0, 0, 0, 0, 0, 0
    all_labels, all_preds = [], []
    n_batches = len(val_dl)
    for step, batch in enumerate(val_dl):
        batch = [t.to(device) for t in batch]
        embeddings, labels = batch
        with torch.no_grad():
            preds = model(embeddings)
            loss = criterion(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            # Compute weighted Cohen's kappa score
            preds = np.argmax(preds, axis=1)
            labels = labels.detach().cpu().numpy()
            kappa = cohen_kappa_score(preds, labels, weights='quadratic')
            v_acc += accuracy_score(labels, preds)
            v_qwk += kappa
            f1_micro += f1_score(labels, preds, average='micro')
            f1_macro += f1_score(labels, preds, average='macro')
            f1_weighted += f1_score(labels, preds, average='weighted')
            all_labels.extend(labels)
            all_preds.extend(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / n_batches
    v_qwk = v_qwk / n_batches
    v_acc = v_acc / n_batches
    f1_micro = f1_micro / n_batches
    f1_macro = f1_macro / n_batches
    f1_weighted = f1_weighted / n_batches
    c_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, v_qwk, v_acc, f1_micro, f1_macro, f1_weighted, c_matrix


def fit(model, criterion, train_data, test_data, epochs, learning_rate):
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=.9)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t_0, t_mult)

    metrics = {'train_loss': [],
               'val_loss': [],
               'train_qwk': [],
               'val_qwk': [],
               'c_matrix': [],
               'val_acc': [],
               'f1_micro': [],
               'f1_macro': [],
               'f1_weighted': [],
               }

    for epoch in range(epochs):
        train_loss, train_qwk, train_acc, _, _, _ = train_epoch(train_data, model, optimizer, criterion, epoch,
                                                                scheduler)
        val_loss, val_qwk, val_acc, f1_micro, f1_macro, f1_weighted, c_matrix = evaluate_epoch(test_data, model,
                                                                                               criterion)
        epoch += 1

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_qwk'].append(train_qwk)
        metrics['val_qwk'].append(val_qwk)
        metrics['val_acc'].append(val_acc)
        metrics['f1_micro'].append(f1_micro)
        metrics['f1_macro'].append(f1_macro)
        metrics['f1_weighted'].append(f1_weighted)
        metrics['c_matrix'].append(c_matrix)

    return metrics


def run_cv(data_x, labels, k, batch_size, epochs, learning_rate):
    kf = StratifiedKFold(n_splits=k, random_state=69420, shuffle=True)

    history = []

    # if reduced_classes:
    #     # Reduce the number of classes to predict
    #     labels[labels == 4] = 3

    for train_index, test_index in kf.split(data_x, y=labels, groups=None):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_data, val_data = init_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)
        class_weights = get_class_weights(y_train)

        weights = torch.tensor(class_weights, dtype=torch.float)
        weights = weights.to(device)

        model = get_classifier(embeddings_size=X_train.shape[-1], n_classes=y_train.unique().shape[0])
        model = model.to(device)

        criterion = nn.NLLLoss(weight=weights)

        metrics = fit(model, criterion, train_data, val_data, epochs, learning_rate)
        history.append(metrics)

    logger.info(
        f"Best average val QWK over {k}-folds: {np.mean(np.max([metric['val_qwk'] for metric in history], axis=1))}")
    logger.info(
        f"Best average accuracy over {k}-folds: {np.mean(np.max([metric['val_acc'] for metric in history], axis=1))}")
    logger.info(
        f"Best average F1-Micro {k}-folds: {np.mean(np.max([metric['f1_micro'] for metric in history], axis=1))}")
    logger.info(
        f"Best average F1-Macro {k}-folds: {np.mean(np.max([metric['f1_macro'] for metric in history], axis=1))}")
    logger.info(
        f"Best average F1-Weighted {k}-folds: {np.mean(np.max([metric['f1_weighted'] for metric in history], axis=1))}")
    # logger.info(
    #     f"Mean val QWK for each fold: {np.mean([metric['val_qwk'] for metric in history], axis=1)}")
    logger.info(f"Best QWK for each fold: {np.max([metric['val_qwk'] for metric in history], axis=1)}")
    logger.info(f"Best QWK for each fold: {np.max([metric['val_qwk'] for metric in history], axis=1)}")
    logger.info(f"Best F1-Micro each fold: {np.max([metric['f1_micro'] for metric in history], axis=1)}")
    logger.info(f"Best F1-Macro each fold: {np.max([metric['f1_macro'] for metric in history], axis=1)}")
    logger.info(f"Best F1-Weighted each fold: {np.max([metric['f1_weighted'] for metric in history], axis=1)}")

    return history


def train_model(data_x, labels, batch_size, epochs, learning_rate):
    history = []

    # if reduced_classes:
    #     labels[labels == 4] = 3

    X_train, X_test, y_train, y_test = train_test_split(data_x, labels, test_size=.3, random_state=69420, shuffle=True,
                                                        stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=69420, shuffle=True,
                                                      stratify=y_train)

    train_data, val_data = init_dataloaders(X_train, y_train, X_val, y_val, batch_size=batch_size)
    class_weights = get_class_weights(y_train)

    weights = torch.tensor(class_weights, dtype=torch.float)
    weights = weights.to(device)

    model = get_classifier(embeddings_size=X_train.shape[-1], n_classes=y_train.unique().shape[0])
    model = model.to(device)

    criterion = nn.NLLLoss(weight=weights)

    metrics = fit(model, criterion, train_data, val_data, epochs, learning_rate)
    history.append(metrics)

    logger.info(f"Best QWK for each fold: {np.max([metric['val_qwk'] for metric in history], axis=1)}")
    logger.info(f"Best F1-Micro each fold: {np.max([metric['f1_micro'] for metric in history], axis=1)}")
    logger.info(f"Best F1-Macro each fold: {np.max([metric['f1_macro'] for metric in history], axis=1)}")
    logger.info(f"Best F1-Weighted each fold: {np.max([metric['f1_weighted'] for metric in history], axis=1)}")

    model.eval()

    preds = model(X_test)
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)

    report = classification_report(y_test, preds, labels=y_test.unique(), target_names=['R0', 'R1', 'R2', 'R3', 'R4'])
    print(report)
    kappa = cohen_kappa_score(preds, y_test, weights='quadratic')
    print(f'Test Kappa: {kappa}')
    return history, model


if __name__ == "__main__":
    args, _ = parse_args()

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{args.filename}.log', 'a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    torch.manual_seed(69420)
    np.random.seed(69420)

    _batch_size = args.batch_size
    _epochs = args.epochs
    _learning_rate = args.learning_rate
    _data_dir = args.data_dir
    _k = args.k
    t_0 = args.t_0
    t_mult = args.t_mult
    filename = args.filename
    reduced_classes = args.reduced_classes

    logger.info(f'Starting execution: {datetime.now().isoformat()}')

    logger.info(
        f'batch_size={_batch_size},'
        f' epochs={_epochs},'
        f' learning_rate={_learning_rate},'
        f' k={_k},'
        f' t_0={t_0},'
        f' t_mult={t_mult}'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, df = get_data(_data_dir, filename)

    # Run k fold cross validation
    # history = run_cv(X, y, _k, _batch_size, _epochs, _learning_rate)
    # plot_confusion_matrix(history)
    history, model = train_model(X, y, _batch_size, _epochs, _learning_rate)
    plot_metrics(history)

    plt.title('Shap summary plot')
    fig, ax = plt.gcf(), plt.gca()