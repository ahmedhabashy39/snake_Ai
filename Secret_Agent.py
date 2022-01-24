import torch
import random
import numpy as np
from collections import deque             #two sided queue
from Smart_Snake_Game import smart_snake, direct, coordinates   #import snake,directions,coordinates point
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000  #100,000
lok4a = 1000  #batch size
Learning_Rate = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # etermines the importance of future rewards 0:1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft when maxlength reaches 100,000
        self.model = Linear_QNet(11, 256, 3)   # model with (input layers , hidden layers , output layers)
        self.trainer = QTrainer(self.model, lr=Learning_Rate, gamma=self.gamma)


    def get_state(self, game):   #get the current state of the game
        head = game.Snake[0]
        point_l = coordinates(head.x - 20, head.y) #coordinates of left step 
        point_r = coordinates(head.x + 20, head.y) #coordinates of right step 
        point_u = coordinates(head.x, head.y - 20) #coordinates of up step 
        point_d = coordinates(head.x, head.y + 20) #coordinates of down step 
        # booleans to check if the current direction equals to direct.directoin
        # only one of them is True and the rest are False
        dir_l = game.direction == direct.left
        dir_r = game.direction == direct.right
        dir_u = game.direction == direct.up
        dir_d = game.direction == direct.down

        state = [
            # Danger straight
            (dir_r and game.collide(point_r)) or 
            (dir_l and game.collide(point_l)) or 
            (dir_u and game.collide(point_u)) or 
            (dir_d and game.collide(point_d)),

            # Danger right
            (dir_u and game.collide(point_r)) or 
            (dir_d and game.collide(point_l)) or 
            (dir_l and game.collide(point_u)) or 
            (dir_r and game.collide(point_d)),

            # Danger left
            (dir_d and game.collide(point_r)) or 
            (dir_u and game.collide(point_l)) or 
            (dir_r and game.collide(point_u)) or 
            (dir_l and game.collide(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) #return the state list in the form of integer matrix

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #add every thing in the game to the deque, popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > lok4a: # if the length of the memory is greater than lok4a_size 
            mini_sample = random.sample(self.memory, lok4a) # take a random sample from memory with sample size = lok4a
        else:
            mini_sample = self.memory  #if the length of memory < lok4a take all the stored memory as sample

        states, actions, rewards, next_states, dones = zip(*mini_sample) # collect each parameter from the sample with its peers
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
#the more games we have the smaller epsilon will be which will continuously decrease the probability of geting
#a random action and start getting predicted moves insted
        
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            #turn the state matrix to a tensor of flat datatype
            state0 = torch.tensor(state, dtype=torch.float)
            # return a tensor of size[1,3] where each element corresponds to the intensity of each direction
            # does so by using the forward function in the model file
            prediction = self.model(state0) 
            # return the max intensity item index
            move = torch.argmax(prediction).item()
            # the max intensity prediction index will equal to 1 and the rest will be 0 ex:[0,1,0]
            final_move[move] = 1

        return final_move


def train():
    record = 0  #score at the beginning of the game
    agent = Agent()
    game = smart_snake()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.next_move(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset_game()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record: # if we have a new high score then save model data
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

          

if __name__ == '__main__':
    train()
    