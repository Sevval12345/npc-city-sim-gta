import numpy as np
import pygame
from sim import collide_grid
from behavior import apply_shot, update_states, collide_rects

W, H = 800, 600
N = 2000
RADIUS = 3.0
WANDER_SPEED = 60.0
FLEE_SPEED = 150.0
HEAR_RADIUS = 150.0
FLEE_TIME = 3.0

# Binalar: [x1, y1, x2, y2]
buildings = np.array([
    [120, 100, 260, 200],
    [450, 80, 620, 180],
    [200, 380, 330, 520],
    [520, 340, 700, 440],
], dtype=np.float64)

rng = np.random.default_rng(42)
pos = rng.uniform([0, 0], [W, H], (N, 2))
angle = rng.uniform(0, 2 * np.pi, N)
vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * WANDER_SPEED
state = np.zeros(N, dtype=np.int8)       # 0 = WANDER, 1 = FLEE
timer = np.zeros(N, dtype=np.float64)

# Numba derleme ısınması
apply_shot(pos.copy(), vel.copy(), state.copy(), timer.copy(), 0.0, 0.0, 1.0, 1.0, 1.0)
update_states(vel.copy(), state.copy(), timer.copy(), 0.01, 1.0)
collide_rects(pos.copy(), vel.copy(), buildings, RADIUS)
collide_grid(pos.copy(), vel.copy(), RADIUS, W, H)

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
running = True
shot_rings = []  # [x, y, kalan_sure]

while running:
    dt = clock.tick(60) / 1000
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            sx, sy = e.pos
            apply_shot(pos, vel, state, timer, float(sx), float(sy),
                       HEAR_RADIUS, FLEE_SPEED, FLEE_TIME)
            shot_rings.append([sx, sy, 0.4])

    update_states(vel, state, timer, dt, WANDER_SPEED)
    pos += vel * dt
    hit_x = (pos[:, 0] < 0) | (pos[:, 0] > W)
    hit_y = (pos[:, 1] < 0) | (pos[:, 1] > H)
    vel[hit_x, 0] *= -1
    vel[hit_y, 1] *= -1
    pos[:, 0] = np.clip(pos[:, 0], 0, W)
    pos[:, 1] = np.clip(pos[:, 1], 0, H)
    collide_grid(pos, vel, RADIUS, W, H)
    collide_rects(pos, vel, buildings, RADIUS)

    screen.fill((30, 30, 30))
    for b in buildings:
        pygame.draw.rect(screen, (70, 80, 100), (b[0], b[1], b[2] - b[0], b[3] - b[1]))
    for i in range(N):
        color = (255, 70, 70) if state[i] == 1 else (220, 220, 220)
        pygame.draw.circle(screen, color, (int(pos[i, 0]), int(pos[i, 1])), int(RADIUS))
    for ring in shot_rings:
        ring[2] -= dt
        r = int(HEAR_RADIUS * (1 - ring[2] / 0.4))
        pygame.draw.circle(screen, (255, 200, 60), (ring[0], ring[1]), max(r, 1), 2)
    shot_rings = [r for r in shot_rings if r[2] > 0]

    fleeing = int(state.sum())
    pygame.display.set_caption(
        f"{N} ajan | FPS: {clock.get_fps():.0f} | kaçan: {fleeing} | Sol tık: silah sesi"
    )
    pygame.display.flip()

pygame.quit()