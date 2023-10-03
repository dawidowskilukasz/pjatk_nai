class Board:

    def __init__(self, game_size):
        self.fields = [(i, j) for i in range(1, game_size + 1) for j in range(1, game_size + 1)]
        self.fields_len = len(str(game_size))

    def display(self):
        for i, field in enumerate(self.fields):
            self._display_field(field)
            if (i + 1) % game_size == 0:
                print()

    def _display_field(self, field):
        print(f"({field[0]:<{self.fields_len}}.{field[1]:{self.fields_len}})", end=" ")

    def cover_fields(self, fields_to_cover):
        for field in fields_to_cover:
            index = self.fields.index(field)
            self.fields[index] = (0, 0)

    def generate_possible_moves(self, move_type):
        possible_moves = [
            [field_one, (field_one[0] + move_type[0], field_one[1] + move_type[1])]
            for field_one in self.fields
            if (not (0 in field_one)) and (field_one[0] + move_type[0], field_one[1] + move_type[1]) in self.fields
        ]
        return possible_moves


class Player:

    def __init__(self, name, move_type):
        self.name = name
        self.move_type = move_type

    def provide_fields(self):
        return [tuple(map(int, input(f"{self.name}, provide field {i + 1} to cover: ").split())) for i in range(2)]


class Game:

    def turn(self, player, board):
        fields_to_cover = player.provide_fields()
        if fields_to_cover in board.generate_possible_moves(player.move_type):
            board.cover_fields(fields_to_cover)

    def play(self, players, board):
        while True:
            for player in players:
                if not board.generate_possible_moves(player.move_type):
                    return player.name
                board.display()
                print()
                self.turn(player, board)
                print()


game_size = 3
board = Board(game_size)
players = [Player("Y", (1, 0)), Player("X", (0, 1))]
game = Game()
loser = game.play(players, board)

print(f"Game terminated! Loser: {loser}")