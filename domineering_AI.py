from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
# import time

class Domineering(TwoPlayerGame):

    def __init__(self, players, board_size):
        self.board_size = board_size
        self.board_size_len = len(str(board_size))
        self.board = [(i, j) for i in range(1, board_size + 1) for j in range(1, board_size + 1)]

        self.players = players
        self.current_player = 1

        players[0].move_type = (1, 0)
        players[1].move_type = (0, 1)
        for player in players:
            player.score = 0

    def possible_moves(self):
        return self._generate_possible_moves(self.player)

    def possible_moves_opponent(self):
        return self._generate_possible_moves(self.opponent)

    def _generate_possible_moves(self, ply):
        generated_possible_moves = [
            [field, (field[0] + ply.move_type[0], field[1] + ply.move_type[1])]
            for field in self.board
            if (not (0 in field)) and (
                field[0] + ply.move_type[0], field[1] + ply.move_type[1]) in self.board
        ]
        return generated_possible_moves

    def make_move(self, move):
        for field in move:
            index = self.board.index(field)
            self.board[index] = (0, 0)
        self.player.score -= len(self.possible_moves_opponent())

    def unmake_move(self, move):
        self.player.score += len(self.possible_moves_opponent())
        for field in move:
            index = (field[0] - 1) * self.board_size + (field[1] - 1)
            self.board[index] = field

    def show(self):
        for i, field in enumerate(self.board):
            self._show_field(field)
            if (i + 1) % self.board_size == 0:
                print()

    def _show_field(self, field):
        print(f"({field[0]:<{self.board_size_len}}.{field[1]:{self.board_size_len}})", end=" ")

    def lose(self):
        return not self.possible_moves()

    def scoring(self):
        return self.player.score - self.opponent.score

    def is_over(self):
        return self.lose()


ai = Negamax(5)
game = Domineering([AI_Player(ai), AI_Player(ai)], 6)
# start = time.time()
game.play()
print("player %d loses" % game.current_player)
# end = time.time()
# print(end - start)
