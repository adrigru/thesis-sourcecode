import pandas as pd
import numpy as np
import argparse
import os
import nltk
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import logging
import sys
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from torch import nn, optim
import torch
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM _MODEL_DIR", '.'))

    return parser.parse_known_args()


def get_data(data_dir, filename):
    """
    Get the training data and convert to tensors
    """
    df = pd.read_csv(os.path.join(data_dir, filename), delimiter='\t')

    return df


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, hn = self.rnn(x, h0.detach())

        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out


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


def get_class_weights(y_train):
    return compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train),
                                y=y_train.tolist())


def init_dataloaders(x_train, y_train, x_val, y_val, batch_size=64):
    train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train).long())
    train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)
    val_data = TensorDataset(torch.tensor(x_val), torch.tensor(y_val).long())
    val_dataloader = DataLoader(val_data, sampler=None, batch_size=batch_size)

    return train_dataloader, val_dataloader


def train_epoch(train_dl, model, optimizer, criterion, epoch, scheduler=None):
    total_loss, qwk, acc, f1_micro, f1_macro, f1_weighted = 0, 0, 0, 0, 0, 0

    model.train()
    # https://arxiv.org/pdf/1608.03983.pdf
    n_batches = len(train_dl)
    for i, batch in enumerate(train_dl):
        batch = [r.to(device) for r in batch]
        embeddings, labels = batch
        model.zero_grad()
        preds = model(embeddings.view(-1, 1, embeddings.shape[-1]))
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
            preds = model(embeddings.view(-1, 1, embeddings.shape[-1]))
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
    # c_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, v_qwk, v_acc, f1_micro, f1_macro, f1_weighted


def fit(model, criterion, train_data, test_data, epochs, learning_rate):
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=.9)
    # optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(params=model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t_0, t_mult)
    # scheduler = None

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
        val_loss, val_qwk, val_acc, f1_micro, f1_macro, f1_weighted = evaluate_epoch(test_data, model, criterion)
        epoch += 1

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_qwk'].append(train_qwk)
        metrics['val_qwk'].append(val_qwk)
        metrics['val_acc'].append(val_acc)
        metrics['f1_micro'].append(f1_micro)
        metrics['f1_macro'].append(f1_macro)
        metrics['f1_weighted'].append(f1_weighted)

    return metrics


def get_doc_2_vec_embeddings(train_tagged, test_tagged):
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=1,
                        min_alpha=0.065,
                        alpha=0.065,
                        seed=69420)

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=1, seed=69420)

    model_dmm.build_vocab([x for x in train_tagged.values])

    model_dbow.build_vocab([x for x in train_tagged.values])

    for epoch in range(1):
        model_dmm.train(train_tagged, total_examples=train_tagged.shape[0], epochs=3)
        model_dbow.train(train_tagged, total_examples=train_tagged.shape[0], epochs=2)
        # model_dmm.alpha -= 0.002
        # model_dmm.min_alpha = model_dmm.alpha
        # model_dbow.alpha -= 0.002
        # model_dbow.min_alpha = model_dbow.alpha

    model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

    ys_train, xs_train = get_vectors(model, train_tagged)
    ys_test, xs_test = get_vectors(model, test_tagged)

    return xs_train, xs_test, ys_train, ys_test


def run_cv(data_x, labels, k, batch_size, epochs, learning_rate):
    kf = StratifiedKFold(n_splits=k, random_state=69420, shuffle=True)

    history = []

    if reduced_classes:
        labels[labels == 4] = 3

    data_x = data_x.to_numpy()
    labels = labels.to_numpy()

    for train_index, test_index in kf.split(data_x, y=labels, groups=None):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        _df_train = pd.DataFrame({'x': X_train, 'y': y_train})
        _df_test = pd.DataFrame({'x': X_test, 'y': y_test})

        train_tagged = _df_train.apply(lambda row: TaggedDocument(words=tokenize_text(row.x), tags=[row.y]), axis=1)
        test_tagged = _df_test.apply(lambda row: TaggedDocument(words=tokenize_text(row.x), tags=[row.y]), axis=1)

        xs_train, xs_test, ys_train, ys_test = get_doc_2_vec_embeddings(train_tagged, test_tagged)

        del _df_train, _df_test
        del train_tagged, test_tagged

        train_data, val_data = init_dataloaders(xs_train, ys_train, xs_test, ys_test, batch_size=batch_size)
        class_weights = get_class_weights(y_train)

        weights = torch.tensor(class_weights, dtype=torch.float)
        weights = weights.to(device)

        model = RNNModel(input_dim=xs_train.shape[-1], hidden_dim=256, layer_dim=1,
                         output_dim=np.unique(ys_train).shape[0])
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

    return history


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


if __name__ == '__main__':
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
        f'reduced_clases={args.reduced_classes},'
        f' k={_k},'
        f' t_0={t_0},'
        f' t_mult={t_mult}'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = get_data(_data_dir, filename)

    df.text_en = df.text_en.apply(str.lower)

    history = run_cv(df.text_en, df.label, _k, _batch_size, _epochs, _learning_rate)

    # plot_metrics(history)
