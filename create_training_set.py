import glob
import chess.pgn
import random


def flip_colors(board):
    def flip_char(ch):
        if ch >= 'a' and ch <= 'z':
            return ch.upper()
        elif ch >= 'A' and ch <= 'Z':
            return ch.lower()
        return ch

    board = '\n'.join(reversed(board.split('\n')))
    return ''.join(flip_char(ch) for ch in board)


def game_yielder():
    for fn in glob.glob('games/*pgn'):
        with open(fn) as fin:
            while True:
                game = chess.pgn.read_game(fin)
                if game is None:
                    break
                yield game


def main():
    boards = []
    for game in game_yielder():
        if game.headers['Result'] not in ('1-0', '0-1'):
            continue
        board = game.board()
        number_of_moves = sum(1 for _ in game.mainline_moves())
        one_but_last = None
        two_but_last = None
        for idx, move in enumerate(game.mainline_moves()):
            if idx == number_of_moves - 1:
                one_but_last = board.copy()
            elif idx == number_of_moves - 2:
                two_but_last = board.copy()
            board.push(move)
        if not board.is_checkmate():
            continue

        board = str(board)
        one_but_last = str(one_but_last)
        two_but_last = str(two_but_last)

        # Switch the game such that white is the color whose turn it is:
        if game.headers['Result'] == '0-1':
            one_but_last = flip_colors(one_but_last)
        else:
            two_but_last = flip_colors(two_but_last)
        boards.append((two_but_last, 0.0))
        boards.append((one_but_last, 1.0))
        if len(boards) % 100 == 0:
            print(len(boards))
        if len(boards) >= 100_000:
            break

    with open('training.txt', 'w') as fout:
        for board, score in boards:
            fout.write('\n')
            fout.write(str(score) + '\n')
            fout.write(board)
            fout.write('\n')


if __name__ == '__main__':
    main()
