import time
import numpy as np
import pygame
from sim import collide_bruteforce

W, H = 800, 600
N = 10000
RADIUS = 3.0

rng = np.random.default_rng(42)
pos = rng.uniform([0, 0], [W, H], (N, 2))
angle = rng.uniform(0, 2 * np.pi, N)
vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * 60

collide_bruteforce(pos.copy(), vel.copy(), RADIUS)  # Numba derleme ısınması

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
running = True
step_ms = 0.0

while running:
    dt = clock.tick(60) / 1000
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    t0 = time.perf_counter()
    pos += vel * dt
    hit_x = (pos[:, 0] < 0) | (pos[:, 0] > W)
    hit_y = (pos[:, 1] < 0) | (pos[:, 1] > H)
    vel[hit_x, 0] *= -1
    vel[hit_y, 1] *= -1
    pos[:, 0] = np.clip(pos[:, 0], 0, W)
    pos[:, 1] = np.clip(pos[:, 1], 0, H)
    collide_bruteforce(pos, vel, RADIUS)
    step_ms = 0.9 * step_ms + 0.1 * (time.perf_counter() - t0) * 1000

    screen.fill((30, 30, 30))
    for p in pos:
        pygame.draw.circle(screen, (220, 220, 220), (int(p[0]), int(p[1])), int(RADIUS))
    pygame.display.set_caption(f"{N} ajan | FPS: {clock.get_fps():.0f} | simülasyon: {step_ms:.2f} ms")
    pygame.display.flip()

pygame.quit()