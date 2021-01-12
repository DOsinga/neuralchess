from collections import defaultdict

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


def encode_board_2d(board):
    board = board.replace(' ', '').split('\n')
    return np.asarray([[one_hot(ch) for ch in line] for line in board])


def build_model_1d():
    board_input = keras.Input(shape=(8, 8, 13))
    board_flat = layers.Flatten()(board_input)
    dense_1 = layers.Dense(512, activation="relu")(board_flat)
    dense_2 = layers.Dense(64, activation="relu")(dense_1)
    score = layers.Dense(1, activation='sigmoid')(dense_2)
    model = keras.Model(inputs=board_input, outputs=score, name='neuralchess')
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(),
    )
    return model


def build_model_2d():
    board_input = keras.Input(shape=(8, 8, 13))
    conv_1 = layers.Conv2D(16, (1, 1), activation='relu')(board_input)
    conv_2 = layers.Conv2D(64, (2, 2), activation='relu')(conv_1)
    pooled_2 = layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = layers.Conv2D(1024, (2, 2), activation='relu')(pooled_2)
    pooled_3 = layers.MaxPooling2D((2, 2))(conv_3)
    board_flat = layers.Flatten()(pooled_3)
    dense_1 = layers.Dense(512, activation="relu")(board_flat)
    dense_2 = layers.Dense(64, activation="relu")(dense_1)
    score = layers.Dense(1, activation='sigmoid')(dense_2)
    model = keras.Model(inputs=board_input, outputs=score, name='neuralchess')
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(),
    )
    return model


def adjust_predicted_scores(predicted_scores, by_idx):
    adjusted_scores = np.zeros(predicted_scores.shape)

    right = 0
    for lst in by_idx.values():
        scores = [predicted_scores[t_idx] for label, t_idx in lst]
        played, *others = scores
        m = max(max(others) * 1.2, played)
        if played > max(others):
            right += 1
        d = (1 - (m - played) / sum(others))
        idx_scores = [m, *(d * s for s in others)]
        for score, (label, t_idx) in zip(idx_scores, lst):
            adjusted_scores[t_idx] = score

    return right, adjusted_scores


def main():
    print('Reading training data')
    training = open('training-boards.txt').read().split('\n\n')
    training = [x.strip().split('\n', 3) for x in training]
    training = [(board, float(score), int(idx), label) for score, idx, label, board in training]
    print('Read', len(training), 'positions, converting to internal representation')
    boards = np.asarray([encode_board_2d(board) for board, score, idx, label in training])
    print('Converted')

    by_idx = defaultdict(list)
    for t_idx, (board, score, idx, label) in enumerate(training):
        by_idx[idx].append((label, t_idx))

    X_train, X_test, y_train, y_test = train_test_split(
        boards, np.asarray([score for board, score, idx, label in training]), test_size=0.25
    )

    model = build_model_2d()
    model.summary()

    print('First training')
    model.fit(
        X_train, y_train, batch_size=256, epochs=5, validation_split=0.1
    )

    for mm in range(200):
        print('>>>', mm)
        predicted_scores = model.predict(boards)
        right, adjusted_scores = adjust_predicted_scores(predicted_scores, by_idx)
        loss = model.evaluate(X_test, y_test, verbose=2)

        print('right:', 100 * right / len(by_idx), 'loss:', loss)

        X_train, X_test, y_train, y_test = train_test_split(
            boards, adjusted_scores, test_size=0.25
        )

        model.fit(X_train, y_train, batch_size=256, epochs=2, validation_split=0.1)

        model.save(f'neuralchess-game-cnn-{mm}.model')


if __name__ == '__main__':
    main()
