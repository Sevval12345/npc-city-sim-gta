import numpy as np
import pygame
from sim import collide_grid
from behavior import (apply_shot_goap, update_states, collide_rects,
                      compute_cover_points, WANDER, FLEE, TO_COVER, HIDE)

W, H = 800, 600
N = 2000
RADIUS = 3.0
WANDER_SPEED = 60.0
FLEE_SPEED = 150.0
HEAR_RADIUS = 150.0
FLEE_TIME = 3.0
HIDE_TIME = 4.0
FLEE_COST = 1.2   # "kaç" planının sabit maliyeti (sn)

# Binalar: [x1, y1, x2, y2]
buildings = np.array([
    [120, 100, 260, 200],
    [450, 80, 620, 180],
    [200, 380, 330, 520],
    [520, 340, 700, 440],
], dtype=np.float64)
nb = len(buildings)

rng = np.random.default_rng(42)
pos = rng.uniform([0, 0], [W, H], (N, 2))
angle = rng.uniform(0, 2 * np.pi, N)
vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * WANDER_SPEED
state = np.zeros(N, dtype=np.int8)
timer = np.zeros(N, dtype=np.float64)
target = np.zeros((N, 2), dtype=np.float64)
cover_pts = np.zeros((nb, 2), dtype=np.float64)
cover_ok = np.zeros(nb, dtype=np.bool_)

# Numba derleme ısınması
compute_cover_points(buildings, 0.0, 0.0, RADIUS, float(W), float(H), cover_pts, cover_ok)
apply_shot_goap(pos.copy(), vel.copy(), state.copy(), timer.copy(), target.copy(),
                buildings, cover_pts, cover_ok, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, RADIUS)
update_states(pos.copy(), vel.copy(), state.copy(), timer.copy(), target.copy(),
              0.01, 1.0, 1.0, 1.0)
collide_rects(pos.copy(), vel.copy(), buildings, RADIUS)
collide_grid(pos.copy(), vel.copy(), RADIUS, W, H)

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
running = True
shot_rings = []        # [x, y, kalan_sure]
markers = []           # siper noktaları (debug), [x, y]
marker_t = 0.0

COLORS = {
    WANDER: (220, 220, 220),
    FLEE: (255, 70, 70),
    TO_COVER: (255, 160, 40),
    HIDE: (80, 200, 255),
}

while running:
    dt = clock.tick(60) / 1000
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            sx, sy = float(e.pos[0]), float(e.pos[1])
            compute_cover_points(buildings, sx, sy, RADIUS, float(W), float(H),
                                 cover_pts, cover_ok)
            apply_shot_goap(pos, vel, state, timer, target, buildings, cover_pts,
                            cover_ok, sx, sy, HEAR_RADIUS, FLEE_SPEED, FLEE_TIME,
                            FLEE_COST, RADIUS)
            shot_rings.append([e.pos[0], e.pos[1], 0.4])
            markers = [(cover_pts[b, 0], cover_pts[b, 1]) for b in range(nb) if cover_ok[b]]
            marker_t = 2.0

    update_states(pos, vel, state, timer, target, dt, WANDER_SPEED, FLEE_SPEED, HIDE_TIME)
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
        pygame.draw.circle(screen, COLORS[int(state[i])],
                           (int(pos[i, 0]), int(pos[i, 1])), int(RADIUS))
    marker_t -= dt
    if marker_t > 0:
        for mx, my in markers:
            pygame.draw.rect(screen, (255, 230, 80), (int(mx) - 4, int(my) - 4, 8, 8), 1)
    for ring in shot_rings:
        ring[2] -= dt
        r = int(HEAR_RADIUS * (1 - ring[2] / 0.4))
        pygame.draw.circle(screen, (255, 200, 60), (ring[0], ring[1]), max(r, 1), 2)
    shot_rings = [r for r in shot_rings if r[2] > 0]

    fleeing = int(np.count_nonzero(state == FLEE))
    cover = int(np.count_nonzero((state == TO_COVER) | (state == HIDE)))
    pygame.display.set_caption(
        f"{N} ajan | FPS: {clock.get_fps():.0f} | kaçan: {fleeing} | siperde: {cover} | Sol tık: silah sesi"
    )
    pygame.display.flip()

pygame.quit()