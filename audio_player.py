import pygame

def play_clip(filename):
    pygame.init()
    pygame.mixer.init()
    sound= pygame.mixer.Sound(filename)

    channel = sound.play()
    while channel.get_busy():
        pygame.time.delay(100)