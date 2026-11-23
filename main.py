import time
import numpy as np
import pygame
from sim import collide_bruteforce, collide_grid

W, H = 800, 600
N = 5000
RADIUS = 3.0

rng = np.random.default_rng(42)
pos = rng.uniform([0, 0], [W, H], (N, 2))
angle = rng.uniform(0, 2 * np.pi, N)
vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * 60

# Numba derleme ısınması
collide_bruteforce(pos.copy(), vel.copy(), RADIUS)
collide_grid(pos.copy(), vel.copy(), RADIUS, W, H)

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
running = True
use_grid = True
step_ms = 0.0

while running:
    dt = clock.tick(60) / 1000
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            use_grid = not use_grid
            step_ms = 0.0

    t0 = time.perf_counter()
    pos += vel * dt
    hit_x = (pos[:, 0] < 0) | (pos[:, 0] > W)
    hit_y = (pos[:, 1] < 0) | (pos[:, 1] > H)
    vel[hit_x, 0] *= -1
    vel[hit_y, 1] *= -1
    pos[:, 0] = np.clip(pos[:, 0], 0, W)
    pos[:, 1] = np.clip(pos[:, 1], 0, H)
    if use_grid:
        collide_grid(pos, vel, RADIUS, W, H)
    else:
        collide_bruteforce(pos, vel, RADIUS)
    step_ms = 0.9 * step_ms + 0.1 * (time.perf_counter() - t0) * 1000

    screen.fill((30, 30, 30))
    for p in pos:
        pygame.draw.circle(screen, (220, 220, 220), (int(p[0]), int(p[1])), int(RADIUS))
    mode = "GRID" if use_grid else "BRUTE"
    pygame.display.set_caption(
        f"[{mode}] {N} ajan | FPS: {clock.get_fps():.0f} | simülasyon: {step_ms:.2f} ms | SPACE: mod değiştir"
    )
    pygame.display.flip()

pygame.quit()