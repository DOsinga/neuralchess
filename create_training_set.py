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
        end_score = 1.0 if game.headers['Result'] == '1-0' else 0

        game_boards = []
        board = game.board()
        number_of_moves = sum(1 for _ in game.mainline_moves())
        for idx, move in enumerate(game.mainline_moves()):
            other_moves_str = []
            if idx % 2 == 0:
                other_moves = [m for m in board.legal_moves if m != move]
                if len(other_moves) > 2:
                    if len(other_moves) > 5:
                        other_moves = random.sample(other_moves, 5)
                    for alter_move in other_moves:
                        board.push(alter_move)
                        other_moves_str.append(str(board))
                        board.pop()

            board.push(move)
            if other_moves_str:
                game_boards.append(
                    (
                        (idx * end_score + (number_of_moves - idx) * 0.5) / number_of_moves,
                        str(board),
                        other_moves_str,
                    )
                )
        if board.is_checkmate():
            boards.extend(game_boards)
        if len(boards) > 1000_000:
            break

    with open('training-boards.txt', 'w') as fout:
        for idx, (score, best_board, other_boards) in enumerate(boards):
            fout.write('\n')
            fout.write(str(score * 1.2) + '\n')
            fout.write(str(idx) + '\n')
            fout.write('Played\n')
            fout.write(best_board)
            fout.write('\n')
            for other_board in other_boards:
                fout.write('\n')
                fout.write(str(score * 0.8) + '\n')
                fout.write(str(idx) + '\n')
                fout.write('Not Played\n')
                fout.write(other_board)
                fout.write('\n')


if __name__ == '__main__':
    main()
