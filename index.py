# Imports
import sys
import os
from tkinter.tix import Tree
from cv2 import rotate
from numpy import true_divide
import visualize
import neat
import pygame
import random
import math

# Configuration
pygame.init()



birdPics = [pygame.image.load("flappybird1.png"),pygame.image.load("flappybird1.png")]

bottomPipeImage = pygame.image.load("pipe.png")
bottomPipeImage = pygame.transform.scale(bottomPipeImage, (80, 300))

topPipeImage = pygame.transform.rotate(bottomPipeImage, 180)

backgroundImage = pygame.transform.scale(pygame.image.load("skyline.png"), (300, 480))

FONT = pygame.font.SysFont("comicsans", 30)
SMALLFONT = pygame.font.SysFont("comicsans", 15)
    
generation=0    

  
class Flapper:
    IMGS = birdPics
    MAX_ROT = 25
    ROT_VEL = 20
    ANIMATION_TIME = 10
    
    def __init__(self, x,y):
        self.x = x
        self.alive_time = 0
        self.y =y
        self.tilt = 0
        self.speed = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
        
    def jump(self):
        self.speed = -5
        self.alive_time = 0 
        self.height = self.y
        
    def move(self):
        self.alive_time+=1
        d = self.speed*self.alive_time +1.5*self.alive_time**2
        
        if d>=16: 
            d = 16
        if d<0:
            d-=2 
        
        self.y = self.y+d
        
        if d<0 or self.y<self.height+50:
            if self.tilt<self.MAX_ROT:
                self.tilt=self.MAX_ROT
            else:
                if self.tilt>=90:
                    self.tilt-=self.ROT_VEL
                    
    def draw(self, screen):
        self.img_count+=1
        if self.img_count<self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count<self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count<self.ANIMATION_TIME*2+1:
            self.img = self.IMGS[0]
            self.img_count = 0
        if self.tilt<=-80:
            self.img = self.IMGS[0]
            self.img_count = self.ANIMATION_TIME*2
    
        rotate_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotate_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        screen.blit(rotate_image, new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    
    def __init__(self,x):
        self.x = x
        self.height = 0
        self.gap = 100
        self.VEL = 5
        self.top = 0
        self.bottom = 0
        
        self.pipe_top = pygame.transform.flip(bottomPipeImage, False, True)
        self.pipe_btm = bottomPipeImage
        
        self.passed = False
        self.set_height()
        
    def set_height(self):
        self.height = random.randrange(100, 300)
        self.top = self.height-self.pipe_top.get_height()
        self.bottom = self.height+self.gap
    
    def move(self):
        self.x-=self.VEL
        
    def draw(self, screen):
        screen.blit(self.pipe_top, (self.x, self.top))
        screen.blit(self.pipe_btm, (self.x, self.bottom))
    
    def collision(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.pipe_top)
        btm_mask = pygame.mask.from_surface(self.pipe_btm)

        top_offset = (self.x-bird.x, self.top-round(bird.y))
        btm_offset = (self.x-bird.x, self.bottom-round(bird.y))
        
        b_point = bird_mask.overlap(btm_mask, btm_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        
        if t_point or b_point:
            return True
        
        return False
  

def run(path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path)

    population = neat.Population(config)
    

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(main,50)

def main(genomes, config):
    nets = []
    ge = []
    flappies = []#Flapper(100, 240)
   
    
    for _ , g in genomes:
        net =neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        flappies.append(Flapper(50, 240))
        g.fitness = 0
        ge.append(g)
        
    clock = pygame.time.Clock()
    fps = 30
    width, height = 300, 480
    screen = pygame.display.set_mode((width, height))
    
    
    

    pipes = [Pipe(300)]
    score = 0
    running = True
    # Game loop.
    while  running:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type ==pygame.QUIT:
                running = False
                pygame.quit()
                quit()
                
                
        #flappy.move()

        pipe_ind = 0
        if len(flappies)>0:
            if len(pipes)>1 and flappies[0].x>pipes[0].x+80:
                pipe_ind = 1
        else:
            global generation
            generation+=1
            run = False
            break

        for x, flappy in enumerate(flappies):
            flappy.move()
            ge[x].fitness+=0.1
            
            
            #flappies.index(flappy)
            output = nets[x].activate((flappy.y, abs(flappy.y-pipes[pipe_ind].height), abs(flappy.y-pipes[pipe_ind].bottom)))

            if output[0]>0.5:
                flappy.jump()
                
        
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, flappy in enumerate(flappies):
                if pipe.collision(flappy):
                    ge[x].fitness-=1
                    flappies.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            
                if not pipe.passed and pipe.x<flappy.x:
                    pipe.passed = True
                    add_pipe = True
            
            if pipe.x+80<0:
                rem.append(pipe)
            
            pipe.move()
                        
        if add_pipe:
            score+=1
            for g in ge:
                g.fitness+=5
                    
        
            pipes.append(Pipe(300))
            add_pipe = False
            
        for r in rem:
            pipes.remove(r)
        for x, flappy in enumerate(flappies):
            if flappy.y+50>=480 or flappy.y<0:
                    flappies.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            

        

        draw_window(screen,flappies, pipes, score, len(flappies), generation)    
    

def draw_window(screen, birds, pipes, score, alive, generation):
    
    scoreboard = FONT.render("Score:"+str(score), 1,"white")
    stats = SMALLFONT.render(str(alive)+" bird(s) alive in generation :"+str(generation), 1,"white")
    
    screen.blit(backgroundImage, (0,0))
    for bird in birds:
        bird.draw(screen)
    
    for pipe in pipes:
        pipe.draw(screen)
        
    screen.blit(scoreboard, (300-10-scoreboard.get_width(), 420))
    screen.blit(stats, (10, 20))

    pygame.display.update() 
    
    
    
    
    

    
    
    
    


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt" )
    run(config_path)




 
    
    

