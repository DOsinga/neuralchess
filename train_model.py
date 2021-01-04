import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

CHARS = {
    '.': 0,
    'B': 1,
    'K': 2,
    'N': 3,
    'P': 4,
    'Q': 5,
    'R': 6,
    'b': 7,
    'k': 8,
    'n': 9,
    'p': 10,
    'q': 11,
    'r': 12,
}


def one_hot(ch):
    res = np.zeros(len(CHARS))
    res[CHARS[ch]] = 1
    return res


def encode_board(board):
    board = board.replace(' ', '').replace('\n', '')
    return np.asarray([one_hot(ch) for ch in board])


def build_model():
    board_input = keras.Input(shape=(64, 13))
    board_flat = layers.Flatten()(board_input)
    dense_1 = layers.Dense(512, activation="relu")(board_flat)
    dense_2 = layers.Dense(64, activation="relu")(dense_1)
    score = layers.Dense(1, activation='sigmoid')(dense_2)
    model = keras.Model(inputs=board_input, outputs=score, name='neuralchess')
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )
    return model


def main():
    training = open('training.txt').read().split('\n\n')
    training = [x.strip().split('\n', 1) for x in training]
    training = [(board, float(score)) for score, board in training]
    boards = np.asarray([encode_board(board) for board, score in training])

    X_train, X_test, y_train, y_test = train_test_split(
        boards, np.asarray([score for board, score in training]), test_size=0.25
    )

    model = build_model()

    history = model.fit(
        X_train, y_train, batch_size=256, epochs=10, validation_split=0.1
    )

    test_scores = model.evaluate(X_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save('neuralchess.model')


if __name__ == '__main__':
    main()
