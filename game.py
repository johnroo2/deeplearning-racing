import random
import pygame
import json
import os
import tensorflow as tf
import numpy as np
from car import Car

clock = pygame.time.Clock()
pygame.display.set_caption("RACE!!!")
pygame.font.init()

font = pygame.font.SysFont('arial', 18)

surface = pygame.display.set_mode((900, 600))
pygame.display.flip()

SURFACE_BG = (225, 225, 225)
car_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "car.png"))
                                .convert_alpha(), (30, 15))
track_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","track.png"))
                                .convert_alpha(), (600, 600))
TRACK_MASK = pygame.mask.from_surface(track_img)
Car.IMG, Car.TRACK_MASK = car_img, TRACK_MASK

with open("settings.json") as settings_data_file:
    settings = json.load(settings_data_file)

@tf.function
def predict(model, probes):
    return model(probes)

class Game: 

    def __init__(self):
        self.generation = 1
        self.max_generation = settings.get('generations')
        self.population = []
        self.draws = []
        self.parents = []
        self.top, self.avg, self.std, self.current_top = 0, 0, 0, 0
        self.frames = 0
        self.run = True
        self.surface = surface

    def render_text(self, text, coords):
        label = font.render(text,1,(0, 0, 0))
        self.surface.blit(label, coords)

    def draw(self):
        self.surface.fill(SURFACE_BG)
        pygame.draw.rect(self.surface, (40, 40, 40), (0, 0, 600, 600))
        self.surface.blit(track_img, (0, 0))
        destroy = []
        for car in self.draws:
            car.move()
            car.draw(surf=self.surface)
            if car.on_wall(): destroy.append(car)  
        for car in destroy:
            self.draws.remove(car)
        self.render_text(f"Gen {self.generation} of {self.max_generation}", (620, 20))
        self.render_text(f"Remaining Cars: {len(self.draws)}", (620, 45))
        self.render_text(f"Time Left: {'{:.2f}'.format((settings.get('max_frames') - self.frames)*0.05)}s", (620, 70))

        if self.frames % 5 == 0:
            pop_fitness = sorted([car.fitness for car in self.population], reverse=True)
            current_fitness = sorted([car.fitness for car in self.draws], reverse=True) if len(self.draws) > 0 else []
            self.top, self.avg, self.std = np.max(pop_fitness), np.mean(pop_fitness), np.std(pop_fitness)
            if len(current_fitness) > 0: self.current_top = np.max(current_fitness)
            
        self.render_text(f"Top Fitness: {'{:.0f}'.format(self.top)}", (620, 110))
        self.render_text(f"Current Top Fitness: {'{:.0f}'.format(self.current_top)}", (620, 135))
        self.render_text(f"Mean Fitness: {'{:.2f}'.format(self.avg)}", (620, 160))
        self.render_text(f"Standard Deviation: {'{:.2f}'.format(self.std)}", (620, 185))

    def create_chromosome(self, model):
        return [tf.Variable(np.random.randn(*w.shape).astype(np.float32)) for w in model.get_weights()]

    def mutate(self, parent, mutation_rate=settings.get("mutation_rate")):
        mutated_weights = []
        
        for w in parent.get_weights():
            mutation = mutation_rate * np.random.randn(*w.shape).astype(np.float32)
            mutated_weights.append(w + mutation) 
        mutated_model = self.create_model()
        mutated_model.set_weights(mutated_weights)
        
        return mutated_model

    def crossover(self, parents, mutation_rate=settings.get("mutation_rate")):
        parents[0] = self.create_chromosome(parents[0])
        parents[1] = self.create_chromosome(parents[1])
        child_weights = []
        
        for p1_weights, p2_weights in zip(parents[0], parents[1]):
            alpha = random.uniform(0, 1)
            child_weight = alpha * p1_weights + (1 - alpha) * p2_weights      
            mutation = mutation_rate * np.random.randn(*p1_weights.shape).astype(np.float32)
            child_weight += mutation   
            child_weights.append(child_weight)
        
        child_model = self.create_model()
        child_model.set_weights(child_weights)
        
        return child_model

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = (6,)))
        model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=2, activation=tf.nn.tanh))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def create_crossover(self):
        children = np.random.choice([i for i in range(len(self.parents))], size=2, replace=False)
        return self.crossover([self.parents[children[0]], self.parents[children[1]]])

    def create_single(self):
        parent = random.choice([i for i in range(len(self.parents))])
        return self.mutate(self.parents[parent])

    def dist(self):
        seed = random.random()
        if seed < settings.get("dist_crossover"):return self.create_crossover()
        elif seed < settings.get("dist_crossover") + settings.get("dist_single"):return self.create_single()
        else:return self.create_model()
        
    def run_neuroevolution(self):
        self.frames = 0

        if len(self.parents) == 0: self.population = [Car(200, 520, self.create_model()) for _ in range(settings.get('population_size'))]
        else:
            self.population = [Car(200, 520, self.dist()) for _ in range(settings.get('population_size'))]

        self.parents = []
        self.draws = self.population[:]

        while self.run and self.frames < settings.get("max_frames") and len(self.draws) > 0:
            clock.tick(50)
            self.frames += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                    pygame.quit()
                    quit()  
            for car in self.draws:
                probes = np.array([car.get_data()])
                pred = predict(car.model, probes)[0]
                if pred[0] > 0.4: car.accelerate()
                elif pred[0] < -0.4: car.decelerate() 
                if pred[1] > 0.4: car.turnright()
                elif pred[1] < -0.4: car.turnleft()
            self.draw()
            pygame.display.update()

        sorted_population = sorted(self.population, key=lambda car:car.fitness, reverse=True)
        self.parents = [car.model for car in 
                            sorted_population[:round(settings.get("population_size") * settings.get("percent_parents"))]]
        self.draws, self.population = [], []

    def init_game(self):
        while self.generation <= self.max_generation:
            self.run_neuroevolution()
            self.generation += 1
        pygame.quit()
        quit()

main_game = Game()