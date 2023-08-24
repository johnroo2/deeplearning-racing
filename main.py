import pygame
from game import main_game

surface = pygame.display.set_mode((900, 600))
pygame.display.flip()

run = True
clock = pygame.time.Clock()
pygame.display.set_caption("RACE!!!")

while run:
    main_game.draw(surface)
    clock.tick(50)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
            quit()    
    pygame.display.update()