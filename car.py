import pygame
import math
import json

with open("settings.json") as settings_data_file:
    settings = json.load(settings_data_file)

class Car:
    IMG = None
    TRACK_MASK = None
    MIN_ROT_VEL = 2.00
    MAX_ROT_VEL = 5.00
    ROT_ACCEL = 0.10

    VEL_ACCEL = 0.05
    MIN_VEL = 4.00
    START_VEL = 5.50
    TERMINAL_VEL = 12.00

    PROBE_DISTANCE = 5.00
    PROBE_COLOUR = (150, 225, 150)

    TOLERANCE = 120

    FITNESS_ACCEL_THRESHOLD_1 = MIN_VEL * settings.get("max_frames") * 0.040
    FITNESS_ACCEL_THRESHOLD_2 = MIN_VEL * settings.get("max_frames") * 0.080

    def __init__(self, x, y, model):
        self.x, self.y = x, y
        self.speed = Car.START_VEL
        self.rot = 0
        self.rot_vel = Car.MIN_ROT_VEL
        self.last_turn = None
        self.img = Car.IMG
        self.model = model
        self.fitness = 0

    def get_probe_from_angle(self, current, x_jump, y_jump, dist, dist_jump):
        collide = Car.TRACK_MASK.get_at(current)
        if collide:
            return (current, dist)
        else:
            return self.get_probe_from_angle((current[0] + x_jump, current[1] + y_jump), 
                                             x_jump, y_jump, dist + dist_jump, dist_jump)
        
    def determine_all_probes(self):
        rotate_img = pygame.transform.rotate(self.img, -self.rot)
        rotate_rect = rotate_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        probe_coords_array = []
        probe_distance_array = []

        for angle in range(-90, 91, 45):
            current_angle = math.radians(self.rot+angle)
            probe_data = self.get_probe_from_angle(rotate_rect.center, 
                        Car.PROBE_DISTANCE * math.cos(current_angle), 
                        Car.PROBE_DISTANCE * math.sin(current_angle), 0, 10)
            probe_coords_array.append(probe_data[0])
            probe_distance_array.append(probe_data[1])
        
        return (probe_coords_array, probe_distance_array)
    
    def get_data(self):
        probes = self.determine_all_probes()[1]
        vectors = [self.speed]
        return [*probes, *vectors]

    def move(self):
        self.fitness += self.speed / 10.0

        self.x += math.cos(math.radians(self.rot)) * self.speed
        self.y += math.sin(math.radians(self.rot)) * self.speed

    def on_wall(self):
        self_mask = pygame.mask.from_surface(self.img)
        collide = Car.TRACK_MASK.overlap_area(self_mask, (self.x, self.y))
        return True if collide and collide > Car.TOLERANCE else False
    
    def decelerate(self): 
        self.speed = max(Car.MIN_VEL, self.speed - Car.VEL_ACCEL)

    def accelerate(self): 
        if self.speed < Car.TERMINAL_VEL: 
            if self.fitness < Car.FITNESS_ACCEL_THRESHOLD_1:
                self.fitness += 1
            elif self.fitness < Car.FITNESS_ACCEL_THRESHOLD_2:
                self.fitness += 2.5
            else:
                self.fitness += 6
        self.speed = min(Car.TERMINAL_VEL, self.speed + 8 * Car.VEL_ACCEL)

    def turnleft(self):
        if self.last_turn != "left":
            self.last_turn = "left"
            self.rot_vel = Car.MIN_ROT_VEL
        self.rot -= self.rot_vel
        self.rot_vel = min(Car.MAX_ROT_VEL, self.rot_vel + Car.ROT_ACCEL * 3)

    def turnright(self):
        if self.last_turn != "right":
            self.last_turn = "right"
            self.rot_vel = Car.MIN_ROT_VEL
        self.rot += self.rot_vel
        self.rot_vel = min(Car.MAX_ROT_VEL, self.rot_vel + Car.ROT_ACCEL * 3)

    def draw(self, surf):
        rotate_img = pygame.transform.rotate(self.img, -self.rot)
        rotate_rect = rotate_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        surf.blit(rotate_img, rotate_rect.topleft)

        probes = self.determine_all_probes()
        for coords in probes[0]:
            pygame.draw.line(surf, Car.PROBE_COLOUR, rotate_rect.center, coords)
            pygame.draw.circle(surf, Car.PROBE_COLOUR, coords, 3)