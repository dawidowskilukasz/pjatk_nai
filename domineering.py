class Board:

    def __init__(self, game_size):
        self.fields = [(i, j) for i in range(1, game_size + 1) for j in range(1, game_size + 1)]
        self.fields_len = len(str(game_size))

    def display(self):
        for i, field in enumerate(self.fields):
            self.display_field(field)
            if (i + 1) % game_size == 0:
                print("")

    def display_field(self, field):
        print(f"({field[0]:<{self.fields_len}}.{field[1]:{self.fields_len}})", end=" ")

    def cover(self, fields_to_cover):
        for i in range(0, 2):
            index = self.fields.index(fields_to_cover[i])
            self.fields[index] = (0, 0)


class Player:

    def __init__(self, name, move_type):
        self.name = name
        self.score = []
        self.move_type = move_type
        self.moves_possible = []

    def generate_possible_moves(self, board):
        self.moves_possible = []

        for field_one in board.fields:
            field_two = (field_one[0] + self.move_type[0], field_one[1] + self.move_type[1])
            if (not (0 in field_one)) and field_two in board.fields:
                self.moves_possible.append([field_one, field_two])


class Game:

    def provide_moves(self, player):

        fields_to_cover = []

        for i in range(0, 2):
            field = input(player.name + ", provide " + str(i+1) + " field to cover: ")
            # adding the element
            fields_to_cover.append(tuple([int(x) for x in field.split(" ")]))

        print(fields_to_cover)
        return fields_to_cover

    def turn(self, player, board):
        fields_to_cover = self.provide_moves(player)
        player.generate_possible_moves(board)

        if fields_to_cover in player.moves_possible:
            board.cover(fields_to_cover)


game_size = 3

board = Board(game_size)
player_Y = Player("Y", (1, 0))
player_Y.generate_possible_moves(board)

player_Y = Player("X", (0, 1))
player_Y.generate_possible_moves(board)

game = Game()

while not (len(player_Y.moves_possible) == 0):
    board.display()
    print()
    game.turn(player_Y, board)
    print()
    player_Y.generate_possible_moves(board)

print("Game terminated!")