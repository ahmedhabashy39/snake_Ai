import pygame
import random                  # To place food randomly on board
import numpy as np
from collections import namedtuple #assign name to each tuple for more readable code
from enum import Enum
pygame.init()                  # To initialize pygame modules

class direct(Enum):       # To simplify using directions  
    right = 1
    left = 2
    up = 3
    down = 4
    
    
# colors
red = (200, 0, 0)
black =(0,0,0) 
white =(255,255,255)
red = (200,0,0)
txtcolor =(192,192,192)
rose =(253,93,168)  
magenta = (255, 0 , 255)
green = (0,255,0)


# font for the text in the game
font = pygame.font.SysFont("SansitaOne.tff",25)
coordinates = namedtuple("coordinates", 'x , y')  # to get points easier
Square = 10   # the size of one square which is the building block of the snake  


class smart_snake:
    #snake game initializer 
    def __init__ (self, width = 600, height = 600):
        self.width = width
        self.height = height 
        
        self.display_screen = pygame.display.set_mode((self.width, self.height)) # display screen
        
        pygame.display.set_caption("Snake Game Terminator AI") #display screen name
        
        self.clock = pygame.time.Clock()  # to control FPS of the game
        
        self.reset_game() #reset function to reset the game after each game
        
        
        
    def put_food (self):
        #get random coordinates for the food
        apple_position = [random.randrange(1, 50)*Square, random.randrange(1, 50)*Square]
        self.food = coordinates(apple_position[0], apple_position[1])
        if self.food in self.Snake:# if the food coordinates is equal to snake then the snake ate it
            self.put_food() # place new food
            
   
    def next_move(self , action):  
        self.frame_iteration += 1
        for move in pygame.event.get():                   # To get any user input
            if move.type == pygame.QUIT:                  # to quit or move the snake
                pygame.quit()                             # up,down,left,right
                quit()
           
                    
        self._move(action)
        self.Snake.insert(0,self.head)  # update the head of the snake       
        
        
        reward = 0
        game_over = False         # checks the collision state and if the snake doesn't improve for too                
        if self.collide() or self.frame_iteration > 100*len(self.Snake): # long it terminates.
            game_over = True
            reward = -10  # To decrease reward and reinforce learning
            return reward , game_over , self.score
        
        
        if self.head == self.food:              # to check when snake eats food and
            self.score += 1                     # increase the score and generate another 
            reward = 10
            self.put_food()                     # food block.
            self.speed += 1                     # increase speed with each apple eaten
        else:
            self.Snake.pop()
        
                    
        
                                            # Displays black screen from the function refresh_ui
        self.refresh_ui()                  # and determines the speed of the snake
        self.clock.tick(self.speed)             # Returns the score and if the game is over 
        game_over = False 
        
        return  reward , game_over , self.score   
        
    def refresh_ui(self):
        self.display_screen.fill(black)
        
        # Draws the snake squares
        for point in self.Snake:
            pygame.draw.circle(self.display_screen,magenta,(point.x,point.y), Square/2)
            pygame.draw.circle(self.display_screen,rose,(point.x+1 , point.y+1) , 3)
        
        # Draws the food    
        pygame.draw.circle(self.display_screen, red, (self.food.x, self.food.y) ,Square/2)        
            
        text = font.render("Score: "+ str(self.score),True,txtcolor) # display text
        self.display_screen.blit(text,[10,10])                      # on screen
        pygame.display.flip()  # update the screen
        
        
    def _move(self, action):     # change the place of the head
        
        movement_clock_wise = [direct.right , direct.down , direct.left , direct.up ]
        index = movement_clock_wise.index(self.direction)
        
        if np.array_equal(action , [1,0,0]):
            new_direction = movement_clock_wise[index] # no change
            
        elif np.array_equal(action , [0,1,0]):     # right(default start) , down , left , up
            next_index = (index + 1) % 4          
            new_direction = movement_clock_wise[next_index] # turn right
            
        else: 
            next_index = (index - 1) % 4               # right , up , left , down 
            new_direction = movement_clock_wise[next_index] # turn left 
            
        self.direction = new_direction    
            
        x = self.head.x
        y = self.head.y
        if self.direction == direct.right:
            x += Square
        elif self.direction == direct.left:
            x -= Square
        elif self.direction == direct.up:
            y -= Square
        elif self.direction == direct.down:
            y += Square    
            
        self.head = coordinates(x, y)    
        
        
        # checks if the snake collided with itself or any wall to end the game    
    def collide(self, point = None):
       if point is None:
            point = self.head
       
       if point.x > self.width - Square or point.x < 0 + Square or point.y > self.height - Square or point.y < 0 + Square:
           return True
           
       if point in self.Snake[1:]:
           return True
       
       return False
   
    
    def reset_game(self):
        self.direction = direct.right                            # initial snake movement
        
        self.head = coordinates(self.width/2 , self.height/2)
        #^^^ The above line is used to initialize the place of the head of the snake 
        # and the lengths are divided by 2 to initialize it at the middle of the game window
        
        self.Snake = [self.head , coordinates(self.head.x - Square, self.head.y),
                      coordinates(self.head.x - (2*Square), self.head.y),
                      coordinates(self.head.x - (3*Square), self.head.y)] 
        #^^^ The snake body initialization
        
        self.score = 0  # Score initialization
         
        self.speed = 80
        
        self.put_food()   # Intitial food position
        
        self.frame_iteration = 0
         
                  
             
        
                      
          