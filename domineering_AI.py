from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import time


class Domineering(TwoPlayerGame):

    def __init__(self, players, board_size):
        self.board_size = board_size
        self.board_size_len = len(str(board_size))
        self.board = {(i, j) for i in range(1, board_size + 1) for j in range(1, board_size + 1)}

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
        x, y = ply.move_type
        return [
            (field, (field[0] + x, field[1] + y))
            for field in self.board if (field[0] + x, field[1] + y) in self.board
        ]

    def make_move(self, move):
        self.board.difference_update(move)
        self.player.score -= len(self.possible_moves_opponent())

    def unmake_move(self, move):
        self.player.score += len(self.possible_moves_opponent())
        self.board.update(move)

    def show(self):
        for i in range(1, self.board_size + 1):
            for j in range(1, self.board_size + 1):
                self._show_field((i, j))
            print()

    def _show_field(self, field):
        if field in self.board:
            print(f"({field[0]:<{self.board_size_len}}.{field[1]:{self.board_size_len}})", end=" ")
        else:
            print(f"({0:<{self.board_size_len}}.{0:{self.board_size_len}})", end=" ")

    def lose(self):
        return not self.possible_moves()

    def scoring(self):
        return self.player.score - self.opponent.score

    def is_over(self):
        return self.lose()


ai = Negamax(15)
game = Domineering([AI_Player(ai), AI_Player(ai)], 5)

start = time.time()

game.play()
print("player %d loses" % game.current_player)

end = time.time()
print(end - start)
