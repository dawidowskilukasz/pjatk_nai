game_size = 8

class Board:

    def __init__(self):
        self.size = game_size
        self.fields = [i for i in range(pow(self.size, 2))]
        self.char_len = len(str(pow(self.size, 2)))

    def display(self):
        for i, val in enumerate(self.fields):
            print(f"{val:^{self.char_len}}", end=" ")
            if (i + 1) % self.size == 0:
                print("")

    def cover(self, field_to_cover):
        self.fields[field_to_cover] = ""


class Player:

    def __init__(self, move_size):
        self.move_size = move_size
        self.score = []

    def move(self):
        move_fields = []
        for i in range(0, 2):
            field = int(input())
            move_fields.append(field)



board = Board()
board.display()
board.cover(45)
board.cover(38)
board.display()

player_1 = Player(8)
player_2 = Player(1)


