from nltk.corpus import stopwords, reuters
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, download as download_dataset
from sklearn.utils import class_weight

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

from random import shuffle
import numpy as np
import time
import pickle

TIME_STR = time.strftime("%Y%m%d-%H%M%S")

params_dict = {
    "NUM_CLASSES": 90,
    "MAX_SEQUENCE_LENGTH": 300,
    "MAX_NB_WORDS": 10000,
    "EMBEDDING_DIM": 64,
    "VALIDATION_SPLIT": 0.1,
    "TEST_SPLIT": 0.2,
    "NUM_LSTM": 450,
    "NUM_DENSE": 225,
    "RATE_DROP_LSTM": 0.3,
    "RATE_DROP_DENSE": 0.3,
    "ACTIVATION_FN": 'relu',
    "PATIENCE": 1,
    "EPOCHS": 100,
    "BATCH_SIZE": 64,
}


def make_data(params):
    """Load reuters datasets, categories, and preprocesses the texts"""
    dataset_filename = "dataset/dataset-TESTSPLIT_{}_NB_WORDS_{}_MAX_SEQ_LENGTH_{}.pkl".format(
        params['TEST_SPLIT'],
        params['MAX_NB_WORDS'],
        params['MAX_SEQUENCE_LENGTH']
    )

    print("Searching for dataset file {} ...".format(dataset_filename))
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
    dataset = []
    for category in reuters.categories():
        for file_id in reuters.fileids(category):
            txt = reuters.raw(file_id)
            txt_lower = txt.lower()
            list_of_words = [stemmer.stem(w) for w in word_tokenize(txt_lower) if w not in stop_words]
            dataset.append((" ".join(list_of_words), categories_to_idx[category]))

    # Shuffle the dataset
    shuffle(dataset)

    # Make train and test sets
    test_limit = int(len(dataset)*params['TEST_SPLIT'])
    test_set = dataset[:test_limit]
    train_set = dataset[test_limit:]

    # Fit the tokenizer on train texts
    tokenizer = Tokenizer(num_words=params['MAX_NB_WORDS'])
    tokenizer.fit_on_texts((txt for txt, category in dataset))

    # Convert them to indices and truncate them if they are too large
    train_sequences = pad_sequences(
        sequences=tokenizer.texts_to_sequences((txt for txt, category in train_set)),
        maxlen=params['MAX_SEQUENCE_LENGTH'])
    test_sequences = pad_sequences(
        sequences=tokenizer.texts_to_sequences((txt for txt, category in test_set)),
        maxlen=params['MAX_SEQUENCE_LENGTH'])
    train_categories = np.array([category for txt, category in train_set])
    test_categories = np.array([category for txt, category in test_set])
    pickle.dump((train_sequences, train_categories, test_sequences, test_categories), open(dataset_filename, "wb"))

    return train_sequences, train_categories, test_sequences, test_categories


def make_model(params):
    """Builds the model"""
    embedding_layer = Embedding(
        input_dim=params['MAX_NB_WORDS'],
        output_dim=params['EMBEDDING_DIM'],
        # weights=[embedding_matrix],
        input_length=params['MAX_SEQUENCE_LENGTH'])
    lstm_layer = LSTM(params['NUM_LSTM'], dropout=params['RATE_DROP_LSTM'], recurrent_dropout=params['RATE_DROP_LSTM'])

    sequence_1_input = Input(shape=(params['MAX_SEQUENCE_LENGTH'],), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    merged = lstm_layer(embedded_sequences_1)

    # sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedded_sequences_2 = embedding_layer(sequence_2_input)
    # y1 = lstm_layer(embedded_sequences_2)
    #
    # merged = concatenate([x1, y1])
    #merged = Dropout(params['RATE_DROP_DENSE'])(x1)  # (merged)
    #merged = BatchNormalization()(merged)

    merged = Dense(params['NUM_DENSE'], activation=params['ACTIVATION_FN'])(merged)
    #merged = Dropout(params['RATE_DROP_DENSE'])(merged)
    #merged = BatchNormalization()(merged)

    preds = Dense(params['NUM_CLASSES'], activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input], outputs=preds)
    model.compile(
        loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc']
    )
    model.summary()
    return model


def train(model, x, y, params):
    early_stopping = EarlyStopping(monitor='val_acc', patience=params['PATIENCE'])
    bst_model_path = '/output/model-{}.h5'.format(TIME_STR)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    hist = model.fit(
        x=x,
        y=np_utils.to_categorical(y),
        validation_split=params['VALIDATION_SPLIT'],
        epochs=params['EPOCHS'],
        batch_size=params['BATCH_SIZE'],
        shuffle=True,
        verbose=2,
        class_weight=class_weights,
        callbacks=[early_stopping, model_checkpoint])

    return hist


def test(model, x, y):
    loss, acc = model.evaluate(x, np_utils.to_categorical(np.array(y)), verbose=0)
    print("Test loss: {:.3f}, test accuracy: {:.3f}".format(loss, acc))
    return loss, acc


def main(params):
    train_sequences, train_categories, test_sequences, test_categories = make_data(params)
    model = make_model(params)

    train(model, train_sequences, train_categories, params)
    loss, acc = test(model, test_sequences, test_categories)

    desc_dict = params.copy()
    desc_dict.update({
        "loss": loss,
        "acc": acc,
    })
    with open("/output/model-{}.txt".format(TIME_STR), 'w') as f:
        desc_str = "{{\n{}\n}}".format("\n".join("  {}: {},".format(k, repr(v)) for k, v in desc_dict.items()))
        f.write(desc_str)
    print(desc_str)

if __name__ == '__main__':
    #main(params_dict)
    make_data(params_dict)
