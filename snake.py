import curses, sys
from random import randint

# TODO: Apple can be generated on the snake

class SnakeGame:
    def __init__(self, board_width=20, board_height=20, gui=False):
        self.score = 0
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui
        self.manual = False

    def start(self):
        self.snake_init()
        self.generate_food()
        if self.gui: self.render_init()
        return self.generate_observations()

    def snake_init(self):
        # Specify the direction constants
        self.DIRECTION_UP = 0
        self.DIRECTION_RIGHT = 1
        self.DIRECTION_DOWN = 2
        self.DIRECTION_LEFT = 3

        x = randint(5, self.board["width"] - 5)
        y = randint(5, self.board["height"] - 5)
        self.snake = []
        self.vertical = randint(0, 1) == 0
        for i in range(3):
            point = [x + i, y] if self.vertical else [x, y + i]
            self.snake.insert(0, point)

        # Set direction so that snake cannot go left while looking right at start
        if self.vertical: self.prev_key = self.DIRECTION_DOWN
        else: self.prev_key = self.DIRECTION_RIGHT

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]
            if food in self.snake: food = []
        self.food = food

    def render_init(self):
        curses.initscr()
        win = curses.newwin(self.board["width"] + 2, self.board["height"] + 2, 0, 0)
        curses.curs_set(0)
        win.nodelay(1)
        win.timeout(200)
        self.win = win
        self.render()

    def render(self):
        self.win.clear()
        self.win.border(0)
        self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
        self.win.addch(self.food[0], self.food[1], 'x')
        for i, point in enumerate(self.snake):
            if i == 0:
                self.win.addch(point[0], point[1], 'O')
            else:
                self.win.addch(point[0], point[1], 'o')
        if not self.manual: self.win.getch()

    def step(self, key):
        if self.done == True: self.end_game()
        # Can't turn right from left and down from up and vice-versa
        # The reason the difference = 2 is because of our choice of directions order
        if abs(key - self.prev_key) == 2: key = self.prev_key

        self.create_new_point(key)
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_last_point()
        self.check_collisions()
        if self.gui: self.render()
        self.prev_key = key
        return self.generate_observations()

    def create_new_point(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == self.DIRECTION_UP:
            new_point[0] -= 1
        elif key == self.DIRECTION_RIGHT:
            new_point[1] += 1
        elif key == self.DIRECTION_DOWN:
            new_point[0] += 1
        elif key == self.DIRECTION_LEFT:
            new_point[1] -= 1
        self.snake.insert(0, new_point)

    def remove_last_point(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        if (self.snake[0][0] == 0 or
            self.snake[0][0] == self.board["width"] + 1 or
            self.snake[0][1] == 0 or
            self.snake[0][1] == self.board["height"] + 1 or
            self.snake[0] in self.snake[1:]):
            self.done = True

    def generate_observations(self):
        return self.done, self.score, self.snake, self.food

    def render_destroy(self):
        curses.endwin()

    def end_game(self):
        if self.gui: self.render_destroy()
        if self.manual: print("Game over. Score: " + str(self.score) + ", args: " + str(args))
        else: raise Exception("Game over")

if __name__ == "__main__":
    game = SnakeGame(gui=True)
    # Read the args to know if the user wants to play manually or not
    args = sys.argv[1:]
    game.manual, steps = False, 20
    if not args: pass
    elif args[0] == '-manual' or args[0] == '-m': game.manual = True
    elif args[0] == '-steps' or args[0] == '-t': steps = int(args[1])

    game.start()
    if game.manual:
        # Need to turn the keypad on in order to be able to read the arrow keys
        game.win.keypad(True)
        key = -1
        while True:
            if game.done: break
            char = game.win.getch()
            if   char == curses.KEY_UP    or char == ord('w'): key = game.DIRECTION_UP
            elif char == curses.KEY_RIGHT or char == ord('d'): key = game.DIRECTION_RIGHT
            elif char == curses.KEY_DOWN  or char == ord('s'): key = game.DIRECTION_DOWN
            elif char == curses.KEY_LEFT  or char == ord('a'): key = game.DIRECTION_LEFT

            # Make a move if the direction is allowed or nothing was pressed
            if key >= game.DIRECTION_UP and key <= game.DIRECTION_LEFT: game.step(key)
        game.end_game()
    else:
        for _ in range(steps):
            game.step(randint(game.DIRECTION_UP, game.DIRECTION_LEFT))
