import curses, sys
from random import randint, shuffle
from collections import defaultdict as ddict

class MazeGame:
    def __init__(self, board_width=20, board_height=20, gui=False):
        self.score = 0
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui
        self.manual = False
        self.move_history = ddict(int)

    def start(self):
        self.game_init()
        self.generate_obstacles()
        if self.gui: self.render_init()
        return self.generate_observations()

    def game_init(self):
        # Specify the direction constants
        self.DIRECTION_UP = 0
        self.DIRECTION_RIGHT = 1
        self.DIRECTION_DOWN = 2
        self.DIRECTION_LEFT = 3

        #x = randint(5, self.board["width"] - 5)
        #y = randint(5, self.board["height"] - 5)
        self.player = [[self.board["width"], 1], [self.board["width"], 0]] # We also need the previous player location for the direction vector
        self.exit   = [1, self.board["height"]]
        self.move_history[str(self.player[0])] += 1
        
        self.prev_key = self.DIRECTION_RIGHT

    def generate_obstacles(self):
        """number_of_obstacles = randint(1000, 2000)
        for _ in range(number_of_obstacles):
            temp_obstacle = []
            while temp_obstacle == []:
                temp_obstacle = [randint(1, self.board["width"]), randint(1, self.board["height"])]
                if temp_obstacle in self.player or temp_obstacle == self.exit: temp_obstacle = []
            self.player.append(temp_obstacle)"""
        # Algorithm taken from http://amertune.blogspot.com/2008/12/maze-generation-in-python.html
        width, height = self.board["height"], self.board["width"]
        # create a list of all walls
        # (all connections between squares in the maze)
        # add all of the vertical walls into the list
        walls = [(x,y,x+1,y)
            for x in range(width-1)
            for y in range(height)]
        # add all of the horizontal walls into the list
        walls.extend([(x,y,x,y+1)
            for x in range(width)
            for y in range(height-1)])

        # create a set for each square in the maze
        cell_sets = [set([(x,y)])
            for x in range(width)
            for y in range(height)]

        # in Kruskal's algorithm, the walls need to be
        # visited in order of weight
        # since we want a random maze, we will shuffle 
        # it and pretend that they are sorted by weight
        walls_copy = walls[:]
        shuffle(walls_copy)

        for wall in walls_copy:
            set_a = None
            set_b = None

            # find the sets that contain the squares
            # that are connected by the wall
            for s in cell_sets:
                if (wall[0], wall[1]) in s:
                    set_a = s
                if (wall[2], wall[3]) in s:
                    set_b = s

            # if the two squares are in separate sets,
            # then combine the sets and remove the 
            # wall that connected them
            if set_a is not set_b:
                cell_sets.remove(set_a)
                cell_sets.remove(set_b)
                cell_sets.append(set_a.union(set_b))
                walls.remove(wall)
        maze = [[1 for _ in range(width)] for _ in range(height)]

        # Indices are all off by one because the board starts with 1 (0 is the border) and the array begins with 0
        maze[self.player[0][0] - 1][self.player[0][1] - 1] = 0
        # Open up the maze for the player
        maze[self.player[0][0] - 2][self.player[0][1] - 1] = 0
        maze[self.player[0][0] - 2][self.player[0][1]] = 0
        maze[self.player[0][0] - 1][self.player[0][1]] = 0

        maze[self.exit[0] - 1][self.exit[1] - 1] = 0
        # Open up the exit
        maze[self.exit[0]][self.exit[1] - 1] = 0
        maze[self.exit[0]][self.exit[1] - 2] = 0
        maze[self.exit[0] - 1][self.exit[1] - 2] = 0
        
        for i in walls:
            maze[i[1]][i[0]] = 0
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == 1:
                    self.player.append([i + 1, j + 1])
        #print(self.player)

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
        self.win.addstr(0, 2, 'Maze. Score: ' + str(self.score) + ' ')
        self.win.addch(self.exit[0], self.exit[1], 'x')
        for i, point in enumerate(self.player):
            if i == 0:
                self.win.addch(point[0], point[1], 'o')
            elif i > 1: # So we don't render previous location
                self.win.addch(point[0], point[1], '█')
        if not self.manual: self.win.getch()

    def step(self, key):
        if self.done == True: self.end_game()
        
        self.move(key)
        if self.move_history[str(self.player[0])] < 4:
            #self.score += 1 # Increase the score by one after every move we survived
            pass
        elif self.move_history[str(self.player[0])] < 8:
            self.score -= 1 # Decrease the score if NN is stuck in an infinite loop
        else:
            self.done = True # Stop the game to avoid the infinite loop
        if self.exit_reached():
            #print("Game won!")
            self.score = 1
            self.done = True
        self.check_collisions()
        if self.gui: self.render()
        self.prev_key = key
        return self.generate_observations()

    def move(self, key):
        self.player[1] = self.player[0][:] # Copy the previous player location for dir. vector
        if key == self.DIRECTION_UP:
            self.player[0][0] -= 1
        elif key == self.DIRECTION_RIGHT:
            self.player[0][1] += 1
        elif key == self.DIRECTION_DOWN:
            self.player[0][0] += 1
        elif key == self.DIRECTION_LEFT:
            self.player[0][1] -= 1
        self.move_history[str(self.player[0])] += 1

    def exit_reached(self):
        return self.player[0] == self.exit

    def check_collisions(self):
        if (self.player[0][0] == 0 or
            self.player[0][0] == self.board["width"] + 1 or
            self.player[0][1] == 0 or
            self.player[0][1] == self.board["height"] + 1 or
            self.player[0] in self.player[2:]): # not self.player[1:], because it contains the previous player position
            self.done = True
            self.player[0] = self.player[1] # This is to show where the player last was before he died

    def generate_observations(self):
        return self.done, self.score, self.player, self.exit

    def render_destroy(self):
        curses.endwin()

    def end_game(self):
        if self.gui: self.render_destroy()
        if self.manual: print("Game over. Score: " + str(self.score) + ", args: " + str(args))
        else: raise Exception("Game over")

if __name__ == "__main__":
    game = MazeGame(gui=True)
    # Read the args to know if the user wants to play manually or not
    args = sys.argv[1:]
    game.manual, steps = False, 0
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
