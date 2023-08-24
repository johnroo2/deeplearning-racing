import pygame
import os
import tensorflow
from car import Car

SURFACE_BG = (225, 225, 225)
car_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "car.png"))
                                .convert_alpha(), (30, 15))
track_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","track.png"))
                                .convert_alpha(), (600, 600))
TRACK_MASK = pygame.mask.from_surface(track_img)
Car.IMG, Car.TRACK_MASK = car_img, TRACK_MASK

class Game: 

    def __init__(self):
        self.cars = []

    def draw_background(self, surf):
        surf.fill(SURFACE_BG)

    def draw(self, surf):
        self.draw_background(surf)
        surf.blit(track_img, (0, 0))

        for car in self.cars:
            car.move()
            car.draw(surf=surf)

main_game = Game()