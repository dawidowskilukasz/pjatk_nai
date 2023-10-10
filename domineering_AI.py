from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import time


'''
Here are basic rules on how to play the game: https://www.byrdseed.com/domineering/; https://en.wikipedia.org/wiki/Domineering
By: Maciej Zagórski(s23575), Łukasz Dawidowski(s22621)
'''
class Domineering(TwoPlayerGame):

    def __init__(self, players, board_size):
        '''
        Initialize Domineering game

        Args:
            players (list): List of two players.
            board_size (int): Size of the board players will play on.
        '''
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
        '''
        Returns possible moves for current player

        Returns:
             list: List of possible moves for player, where every move looks like this (field1, field2).
        '''
        return self._generate_possible_moves(self.player)

    def possible_moves_opponent(self):
        '''
        Returns possible moves for opponent player.

        Returns:
            list: List of possible moves for opponent, where every move looks like this (field1, field2).
        '''
        return self._generate_possible_moves(self.opponent)

    def _generate_possible_moves(self, ply):
        '''
        Generate possible moves for defined player.

        Args:
            ply: Defined player which moves are generated for.

        Returns:
            list: List of possible moves for defined player, where every move looks like this (field1, field2).
        '''
        x, y = ply.move_type
        return [
            (field, (field[0] + x, field[1] + y))
            for field in self.board if (field[0] + x, field[1] + y) in self.board
        ]

    def make_move(self, move):
        '''
        Makes move on the board, AI uses it to decide which move from possible moves is the best.

        Args:
            move (set): Move to make, where move looks like this (field1, field2)
        '''
        self.board.difference_update(move)
        self.player.score -= len(self.possible_moves_opponent())

    def unmake_move(self, move):
        '''
        Undo the move on board, AI uses it to go back a move if it decided move it made is not good enough or want to
        check other possible moves.

        Args:
             move (set): Move to undo, where move looks like this (field1, field2).
        '''
        self.player.score += len(self.possible_moves_opponent())
        self.board.update(move)

    def show(self):
        '''
        Displays in console how the board looks.
        '''
        for i in range(1, self.board_size + 1):
            for j in range(1, self.board_size + 1):
                self._show_field((i, j))
            print()

    def _show_field(self, field):
        '''
        Shows board field.

        Args:
            field (tuple): Pair representing field ond board in form of (x, y).
        '''
        if field in self.board:
            print(f"({field[0]:<{self.board_size_len}}.{field[1]:{self.board_size_len}})", end=" ")
        else:
            print(f"({0:<{self.board_size_len}}.{0:{self.board_size_len}})", end=" ")

    def lose(self):
        '''
        Checks if current player lose.

        Return:
            bool: True, if playes lost; False if player still can make move.
        '''
        return not self.possible_moves()

    def scoring(self):
        '''
        Counts differences in players scores, AI uses it for Deciding which move is better for it.

        Returns:
            int: Current result of current player score subtracted from opponent score.
        '''
        return self.player.score - self.opponent.score

    def is_over(self):
        '''
        Checks if the game is over.

        Returns:
            bool: True if game is finished, False if game is still ongoing
        '''
        return self.lose()


ai = Negamax(15)
game = Domineering([Human_Player(), AI_Player(ai)], 5)

start = time.time()

game.play()
print("player %d loses" % game.current_player)

end = time.time()
print(end - start)
