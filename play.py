import chess
from tensorflow import keras

from train_model import encode_board

model = None

def eval_board(board):
    board = str(board)
    encoded = encode_board(board)
    return model.predict(encoded.reshape(1, 64, 13))[0][0]


def best_move(board, verbose=False):
    max_val = 0
    max_move = None
    for m in board.legal_moves:
        board.push(m)
        score = eval_board(board)
        if verbose:
            print(m, score)
        if score > max_val:
            max_val = score
            max_move = m
        board.pop()
    return max_move, max_val

def main():
    global model
    model = keras.models.load_model('neuralchess.model')
    board = chess.Board()
    while True:
        move, score = best_move(board)
        board.push(move)
        print(board)
        while True:
            move = input('Your move:')
            try:
                board.push_san(move)
                break
            except ValueError as err:
                print(err)

if __name__ == '__main__':
    main()
