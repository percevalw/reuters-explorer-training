import os
import pickle
import time
import warnings
from random import shuffle

import numpy as np
import sklearn.exceptions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize, download as download_dataset
from nltk.corpus import stopwords, reuters
from nltk.stem import SnowballStemmer
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from metrics import *

print(os.environ)

TIME_STR = time.strftime("%Y%m%d-%H%M%S")
PATH_PREFIX = "/"
bst_model_path = PATH_PREFIX + 'output/model-{}.h5'.format(TIME_STR)

params_dict = {
    "NUM_CLASSES": 90,
    "MAX_SEQUENCE_LENGTH": 300,
    "MAX_NB_WORDS": 10000,
    "EMBEDDING_DIM": 100,
    "VALIDATION_SPLIT": 0.2,
    "NUM_LSTM": 300,
    "NUM_DENSE": 200,
    "RATE_DROP_LSTM": 0.2,
    "RATE_DROP_DENSE": 0.2,
    "ACTIVATION_FN": 'relu',
    "PATIENCE": 2,
    "EPOCHS": 100,
    "BATCH_SIZE": 64,
    "LR": 0.001,
}


def val_metrics(y_true, y_pred):
    return {
        'prec': precision(y_true, y_pred),
        'rec': recall(y_true, y_pred),
        'f1': fbeta_score(y_true, y_pred, 1)
    }


def make_data(params):
    """Load reuters datasets, categories, and preprocesses the texts"""
    dataset_filename = PATH_PREFIX + "dataset/dataset-NB_WORDS_{}_MAX_SEQ_LENGTH_{}.pkl".format(
        params['MAX_NB_WORDS'],
        params['MAX_SEQUENCE_LENGTH']
    )

    print("Searching for dataset file {}".format(dataset_filename))
    try:
        file_content = pickle.load(open(dataset_filename, "rb"))
    except (OSError, IOError):
        print("File not found")
    else:
        print("File found")
        return file_content

    download_dataset('stopwords')
    download_dataset('reuters')
    download_dataset('punkt')

    categories_to_idx = {c: i for i, c in enumerate(reuters.categories())}

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer('english')
    train_set = []
    test_set = []
    for file_id in reuters.fileids():
        txt = reuters.raw(file_id)
        txt_lower = txt.lower()
        list_of_words = [stemmer.stem(w) for w in word_tokenize(txt_lower) if w not in stop_words]
        val = (" ".join(list_of_words),
               [categories_to_idx[c] for c in reuters.categories(file_id)],
               file_id)
        if "train" in file_id:
            train_set.append(val)
        else:
            test_set.append(val)

    print("Train set length: {}, test set length: {}".format(len(train_set), len(test_set)))

    mlb = MultiLabelBinarizer(list(range(len(categories_to_idx))))

    # Fit the tokenizer on train texts
    tokenizer = Tokenizer(num_words=params['MAX_NB_WORDS'])
    tokenizer.fit_on_texts((txt for txt, category, _ in train_set))

    # Convert them to indices and truncate them if they are too large
    train_sequences = pad_sequences(
        sequences=tokenizer.texts_to_sequences((txt for txt, category, _ in train_set)),
        maxlen=params['MAX_SEQUENCE_LENGTH'])
    test_sequences = pad_sequences(
        sequences=tokenizer.texts_to_sequences((txt for txt, category, _ in test_set)),
        maxlen=params['MAX_SEQUENCE_LENGTH'])
    train_categories = mlb.fit_transform([categories for txt, categories, _ in train_set])
    test_categories = mlb.fit_transform([categories for txt, categories, _ in test_set])
    train_fileids = [fileid for txt, categories, fileid in train_set]
    test_fileids = [fileid for txt, categories, fileid in test_set]
    pickle.dump((train_sequences, train_categories, test_sequences, test_categories, train_fileids, test_fileids, tokenizer.word_index),
                open(dataset_filename, "wb"))

    return train_sequences, train_categories, test_sequences, test_categories, train_fileids, test_fileids, tokenizer.word_index


def make_model(params, word_index):
    """Builds the model"""
    embedding_matrix = make_embedding_weights(params, word_index)

    embedding_layer = Embedding(
        input_dim=params['MAX_NB_WORDS'],
        output_dim=params['EMBEDDING_DIM'],
        weights=[embedding_matrix],
        input_length=params['MAX_SEQUENCE_LENGTH'])
    lstm_layer = LSTM(params['NUM_LSTM'], dropout=params['RATE_DROP_LSTM'], recurrent_dropout=params['RATE_DROP_LSTM'])

    sequence_1_input = Input(shape=(params['MAX_SEQUENCE_LENGTH'],), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    merged = lstm_layer(embedded_sequences_1)

    merged = Dense(params['NUM_DENSE'], activation=params['ACTIVATION_FN'])(merged)
    merged = Dropout(params['RATE_DROP_DENSE'])(merged)

    preds = Dense(params['NUM_CLASSES'], activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input], outputs=preds)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(lr=params['LR']),
        metrics=['acc', precision, recall, f1]
    )
    model.summary()
    return model


def train(model, x, y, params):
    early_stopping = EarlyStopping(monitor='val_loss', patience=params['PATIENCE'])
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    #class_weights = class_weight.compute_class_weight('balanced', np.arange(params['NUM_CLASSES']), y)

    hist = model.fit(
        x=x,
        y=y,
        validation_split=params['VALIDATION_SPLIT'],
        epochs=params['EPOCHS'],
        batch_size=params['BATCH_SIZE'],
        shuffle=True,
        verbose=2,
        #class_weight=class_weights,
        callbacks=[early_stopping, model_checkpoint])

    return hist


def test(model, x, y):
    model.load_weights(bst_model_path)
    y_prob = model.predict(x, verbose=0)
    y_pred = np.zeros(y_prob.shape, dtype=np.float32)
    y_pred[y_prob > 0.5] = 1

    acc = accuracy_score(y, y_pred)
    loss = log_loss(y, y_pred)
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    print("Best test | loss: {:.3f}, acc: {:.3f}, prec: {:.3f}, rec: {:.3f}, f1: {:.3f}"\
          .format(loss, acc, prec, rec, f1))
    return loss, acc, prec, rec, f1


def make_embedding_weights(params, word_index):
    embeddings_index = {}
    with open(PATH_PREFIX + 'glove/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((params['MAX_NB_WORDS'], params['EMBEDDING_DIM']))
    for word, i in word_index.items():
        if i < params['MAX_NB_WORDS']:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def main(params):
    train_sequences, train_categories, test_sequences, test_categories, train_fileids, test_fileids, word_index = make_data(params)
    model = make_model(params, word_index)

    train(model, train_sequences, train_categories, params)
    loss, acc, prec, rec, f1 = test(model, test_sequences, test_categories)

    desc_dict = params.copy()
    desc_dict.update({
        "metrics_acc": acc,
        "metrics_loss": loss,
        "metrics_prec": prec,
        "metrics_rec": rec,
        "metrics_f1": f1,
    })
    desc_path = PATH_PREFIX + "output/model-{}-{:.3f}.txt".format(TIME_STR, acc)

    with open(desc_path, 'w') as f:
        desc_str = "{{\n{}\n}}\n".format("\n".join(sorted("  {}: {},".format(repr(k), repr(v)) for k, v in desc_dict.items())))
        f.write(desc_str)
        model.summary(print_fn=lambda *args, **kwargs: f.write(*args, **kwargs))
    print(desc_str)

if __name__ == '__main__':
    main(params_dict)
    #make_data(params_dict)
