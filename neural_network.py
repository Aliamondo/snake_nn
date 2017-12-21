from snake import SnakeGame
from maze import MazeGame
from random import randint
import numpy as np
import tflearn
import math
import sys
import os
import pickle
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
from tensorflow.python.framework import ops #This is to fix the visualize issue

class SnakeNN:
    def __init__(self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'snake_nn.tflearn', game_type = 'snake'):
        self.initial_games = initial_games
        print("Initial games number: " + str(self.initial_games))
        self.test_games = test_games
        print("Test games number:    " + str(self.test_games))
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.game_type = game_type
        self.vectors_and_keys = [
                [[-1, 0], 0], # UP
                [[0, 1], 1],  # RIGHT
                [[1, 0], 2],  # DOWN
                [[0, -1], 3]  # LEFT
                ]

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame()
            if self.game_type == 'maze':
                game = MazeGame()
            _, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food  = game.step(game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1]) # Snake is dead
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1]) # The last move was efficient
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0]) # The last move was not efficient
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
        return training_data

    def generate_action(self, snake):
        action = randint(0,2) - 1
        if self.game_type == 'maze':
            action = randint(0,3) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        elif action == 2: # Turn back in a maze game
            new_direction = self.reflect_vector(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21 # This is the board limit

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def reflect_vector(self, vector):
        #print("Reflected the vector (turned around)")
        return np.array([-vector[0], -vector[1]])

    def get_angle(self, a, b):
        # We should first normalize vectors, so that the angle is a value between -1 and 1
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu') # 25 hidden neurons, Rectified Linear Unit [f(x) = max(0, x)]
        #network = fully_connected(network, 100, activation='relu')
        #network = fully_connected(network, 100, activation='relu')
        #network = fully_connected(network, 100, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log' + str(self.initial_games) + "/", tensorboard_verbose=3)
        # To run tensorboard: python3 /home/andrey/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir=log1000/
        #model = tflearn.DNN(network)
        if os.path.isfile(self.filename + ".meta") and os.path.isfile(self.filename + ".index"):
            print("Model file was found")
            model.load(self.filename)
        else:
            print("There was an issue reading the model file, new one will be created")
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1) # Reshape the array so that it is not continuous array anymore, but an array of lists of size 5
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        count = 0
        solved = 0
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame()
            if self.game_type == 'maze':
                game = MazeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    if self.game_type == 'maze' and score == 1: solved += 1
                    count += 1
                    if count % 100 == 0:
                        print('-----')
                        print('id: ' + str(count))
                        print(steps)
                        print(snake)
                        print(food)
                        print(prev_observation)
                        print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        #print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        #print(Counter(scores_arr))
        scores_arr.sort()
        print('Lowest score:',scores_arr[0])
        print('Highest score:',scores_arr[-1])
        if self.game_type == 'maze': print('Total solved mazes:',solved)
        with open('steps_arr', 'wb') as file:
            pickle.dump(steps_arr, file)
        with open('scores_arr', 'wb') as file:
            pickle.dump(scores_arr, file)

    def visualise_game(self, model, game_type):
        game = SnakeGame(gui = True)
        if game_type == 'maze':
            game = MazeGame(gui = True)
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            predictions = []
            for action in range(-1, 2):
               predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(predictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self, game_var):
        ops.reset_default_graph()
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model, game_var)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    args = sys.argv[1:]
    #args=['-v', 'maze']
    #args=['-v']
    if not args: SnakeNN().train()
    elif args[0] == '-visualize' or args[0] == '-v':
        if len(args) == 1: args += ['snake']
        if args[1] == 'maze':
            SnakeNN().visualise('maze')
        else:
            SnakeNN().visualise('snake')
    elif args[0] == '-test':
        game_type = 'snake'
        test_num = 1000
        if len(args) >= 2:
            if args[1] == 'maze':
                game_type = 'maze'
            if len(args) >= 3:
                test_num = int(args[2])
        SnakeNN(test_games = test_num, game_type = game_type).test()
    elif args[0] == '-train' or args[0] == '-t':
        game_type = 'snake'
        if len(args) >= 2:
            if args[1] == 'maze':
                game_type = 'maze'
            if len(args) >= 3:
                training_num = int(args[2])
                test_num = 1000
                if len(args) >= 4:
                    test_num = int(args[3])
        SnakeNN(initial_games = training_num, test_games = test_num, game_type = game_type).train()
