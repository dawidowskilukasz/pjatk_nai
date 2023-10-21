"""
Domineering

Installation:
Assuming that you have pip installed, type this in a terminal: sudo pip install easyAI

Rules:
Here are basic rules on how to play the Domineering game: two players take turns to place lines (tiles) on the board,
covering up fields. One cover two fields vertically, the other covers two fields horizontally. The first player who
cannot move (who cannot cover two fields in a specific way) loses.

Authors:
By Maciej Zagórski (s23575) and Łukasz Dawidowski (s22621), group 72c (10:15-11:45)

Sources:
https://www.byrdseed.com/domineering/ , https://en.wikipedia.org/wiki/Domineering (Domineering rules)
https://zulko.github.io/easyAI/index.html , https://github.com/Zulko/easyAI (easyAI documentation)
"""

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax


class Domineering(TwoPlayerGame):

    def __init__(self, players, board_size):
        """
        Initializes the Domineering game.

        Args:
            players (list): List of two players.
            board_size (int): Size of the board on which the players will play on.
        """
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
        """
        Returns possible moves for the current player.

        Returns:
             list: List of possible moves for the player, where every move looks like this {field1, field2}.
        """
        return self._generate_possible_moves(self.player)

    def possible_moves_opponent(self):
        """
        Returns possible moves for the other player (opponent).

        Returns:
            list: List of possible moves for the opponent, where every move looks like this {field1, field2}.
        """
        return self._generate_possible_moves(self.opponent)

    def _generate_possible_moves(self, ply):
        """
        Generate possible moves for the defined player.

        Args:
            ply: The defined player which moves are generated for.

        Returns:
            list: List of possible moves for the defined player, where every move looks like this {field1, field2}
            (depending on the player's move type the move is horizontal or vertical).
        """
        x, y = ply.move_type
        return [
            (field, (field[0] + x, field[1] + y))
            for field in self.board if (field[0] + x, field[1] + y) in self.board
        ]

    def make_move(self, move):
        """
        Makes move on the board; AI uses it to decide which move from the possible moves is the best (the fewer possible
        moves for the opponent, the better move for the player).

        Args:
            move (set): Move to make, where the move looks like this {field1, field2}.
        """
        self.board.difference_update(move)
        self.player.score -= len(self.possible_moves_opponent())

    def unmake_move(self, move):
        """
        Undo the move on board; AI will use it to undo the move if it decides that the move it made is not good enough
        or want to check other possible moves. Provided to for speeding up the game, based on:
        https://zulko.github.io/easyAI/speedup.html#use-unmake-move

        Args:
             move (set): Move to undo, where the move looks like this {field1, field2}.
        """
        self.player.score += len(self.possible_moves_opponent())
        self.board.update(move)

    def show(self):
        """
        Displays in the console how the board (the current state of the game) looks.
        """
        for i in range(1, self.board_size + 1):
            for j in range(1, self.board_size + 1):
                self._show_field((i, j))
            print()

    def _show_field(self, field):
        """
        Shows board field.

        Args:
            field (tuple): Pair representing field on the board in the form of (X-position, Y-position).
        """
        mark_x, mark_y = field if field in self.board else (0, 0)
        print(f"({mark_x:<{self.board_size_len}}.{mark_y:{self.board_size_len}})", end=" ")

    def lose(self):
        """
        Checks if the current player has lost.

        Return:
            bool: True, if the player cannot make a move.
        """
        return not self.possible_moves()

    def scoring(self):
        """
        Counts differences in the players' scores; AI uses it for deciding which move is better.

        Returns:
            int: Result of the current player's score reduced by the opponent's score (considered as penalty points).
        """
        return self.player.score - self.opponent.score

    def is_over(self):
        """
        Checks if the game is over.

        Returns:
            bool: True if the player has lost, false if not (if the player can make a move).
        """
        return self.lose()


ai = Negamax(5)
game = Domineering([AI_Player(ai), AI_Player(ai)], 5)
# game = Domineering([Human_Player(), AI_Player(ai)], 5)
game.play()
print("player %d loses" % game.current_player)