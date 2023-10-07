from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax


class Domineering(TwoPlayerGame):

    def __init__(self, players, board_size):
        self.players = players
        self.current_player = 1

        self.board_size = board_size
        self.board_size_len = len(str(board_size))
        self.board = [(i, j) for i in range(1, board_size + 1) for j in range(1, board_size + 1)]

        players[0].move_type = (1, 0)
        players[1].move_type = (0, 1)

    def possible_moves(self):
        possible_moves = [
            [field, (field[0] + self.player.move_type[0], field[1] + self.player.move_type[1])]
            for field in self.board
            if (not (0 in field)) and (
                field[0] + self.player.move_type[0], field[1] + self.player.move_type[1]) in self.board
        ]
        return possible_moves

    def make_move(self, move):
        for field in move:
            index = self.board.index(field)
            self.board[index] = (0, 0)

    def show(self):
        for i, field in enumerate(self.board):
            print(f"({field[0]:<{self.board_size_len}}.{field[1]:{self.board_size_len}})", end=" ")
            if (i + 1) % self.board_size == 0:
                print()

    def lose(self):
        return self.possible_moves() == []

    def scoring(self):
        return -1000 if (self.possible_moves() == []) else 0

    def is_over(self):
        return self.lose()


ai = Negamax(8)
game = Domineering([Human_Player(), AI_Player(ai)], 5)
game.play()
print("player %d loses" % (game.current_player))
