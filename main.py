import os
import time

import numpy as np
import pygame
from PIL import Image

from sim import collide_grid
from behavior import (apply_shot_goap, update_states, collide_rects,
                      compute_cover_points, compute_fields, build_blocked,
                      WANDER, FLEE, TO_COVER, HIDE)

W, H = 800, 600
N = 2000
RADIUS = 3.0
WANDER_SPEED = 60.0
FLEE_SPEED = 150.0
HEAR_RADIUS = 150.0
FLEE_TIME = 3.0
HIDE_TIME = 4.0
FLEE_COST = 1.2        # "kaç" planının sabit maliyeti (sn)
CELL = 10.0            # yol bulma ızgarası hücre boyutu (px)
COVER_OFFSET = 18.0    # siper noktasının bina kenarına uzaklığı (px)

# GIF kaydı ayarları
REC_EVERY = 4
REC_MAX = 180
REC_SIZE = (480, 360)

# Binalar: [x1, y1, x2, y2]
buildings = np.array([
    [120, 100, 260, 200],
    [450, 80, 620, 180],
    [200, 380, 330, 520],
    [520, 340, 700, 440],
], dtype=np.float64)
nb = len(buildings)

GW = int(W // CELL)
GH = int(H // CELL)
blocked = np.zeros((GH, GW), dtype=np.bool_)
build_blocked(buildings, CELL, blocked)
dist = np.full((nb, GH, GW), 1e9, dtype=np.float64)

rng = np.random.default_rng(42)
pos = rng.uniform([0, 0], [W, H], (N, 2))
angle = rng.uniform(0, 2 * np.pi, N)
vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * WANDER_SPEED
state = np.zeros(N, dtype=np.int8)
timer = np.zeros(N, dtype=np.float64)
goal = np.zeros(N, dtype=np.int32)
cover_pts = np.zeros((nb, 2), dtype=np.float64)
cover_ok = np.zeros(nb, dtype=np.bool_)

# Numba derleme ısınması
compute_cover_points(buildings, 0.0, 0.0, float(W), float(H), COVER_OFFSET, cover_pts, cover_ok)
compute_fields(blocked, cover_pts, cover_ok, dist, CELL)
apply_shot_goap(pos.copy(), vel.copy(), state.copy(), timer.copy(), goal.copy(), dist,
                blocked, cover_pts, cover_ok, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, CELL)
update_states(pos.copy(), vel.copy(), state.copy(), timer.copy(), goal.copy(), dist,
              blocked, cover_pts, cover_ok, CELL, 0.01, 1.0, 1.0, 1.0)
collide_rects(pos.copy(), vel.copy(), buildings, RADIUS)
collide_grid(pos.copy(), vel.copy(), RADIUS, W, H)
cover_ok[:] = False  # ısınma verisini temizle

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
running = True
shot_rings = []
markers = []
marker_t = 0.0
show_grid = False

# engelli ızgara hücreleri (G tuşuyla görünür)
grid_surf = pygame.Surface((W, H), pygame.SRCALPHA)
for gy in range(GH):
    for gx in range(GW):
        if blocked[gy, gx]:
            pygame.draw.rect(grid_surf, (255, 80, 80, 70),
                             (int(gx * CELL), int(gy * CELL), int(CELL), int(CELL)), 1)

recording = False
rec_frames = []
frame_no = 0


def save_gif(frames):
    os.makedirs("docs", exist_ok=True)
    frames[0].save("docs/demo.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / 15), loop=0, optimize=True)
    print(f"docs/demo.gif kaydedildi ({len(frames)} kare)")


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
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_g:
            show_grid = not show_grid
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            if not recording:
                recording = True
                rec_frames = []
                frame_no = 0
                print("Kayıt başladı (durdurmak için tekrar R)")
            else:
                recording = False
                if rec_frames:
                    save_gif(rec_frames)
                rec_frames = []
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            sx, sy = float(e.pos[0]), float(e.pos[1])
            t0 = time.perf_counter()
            compute_cover_points(buildings, sx, sy, float(W), float(H), COVER_OFFSET,
                                 cover_pts, cover_ok)
            compute_fields(blocked, cover_pts, cover_ok, dist, CELL)
            t1 = time.perf_counter()
            heard, covered = apply_shot_goap(pos, vel, state, timer, goal, dist, blocked,
                                             cover_pts, cover_ok, sx, sy, HEAR_RADIUS,
                                             FLEE_SPEED, FLEE_TIME, FLEE_COST, CELL)
            t2 = time.perf_counter()
            print(f"Mesafe alanları: {(t1 - t0) * 1000:.2f} ms | plan seçimi: "
                  f"{(t2 - t1) * 1000:.2f} ms | duyan: {heard} | siper seçen: {covered}")
            shot_rings.append([e.pos[0], e.pos[1], 0.4])
            markers = [(cover_pts[b, 0], cover_pts[b, 1]) for b in range(nb) if cover_ok[b]]
            marker_t = 2.0

    update_states(pos, vel, state, timer, goal, dist, blocked, cover_pts, cover_ok,
                  CELL, dt, WANDER_SPEED, FLEE_SPEED, HIDE_TIME)
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
    if show_grid:
        screen.blit(grid_surf, (0, 0))
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

    if recording:
        frame_no += 1
        if frame_no % REC_EVERY == 0:
            raw = pygame.image.tostring(screen, "RGB")
            img = Image.frombytes("RGB", (W, H), raw).resize(REC_SIZE)
            rec_frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=32))
            if len(rec_frames) >= REC_MAX:
                recording = False
                save_gif(rec_frames)
                rec_frames = []

    fleeing = int(np.count_nonzero(state == FLEE))
    cover = int(np.count_nonzero((state == TO_COVER) | (state == HIDE)))
    rec_tag = " | KAYIT" if recording else ""
    pygame.display.set_caption(
        f"{N} ajan | FPS: {clock.get_fps():.0f} | kaçan: {fleeing} | siperde: {cover}"
        f" | Sol tık: silah sesi | R: GIF | G: ızgara{rec_tag}"
    )
    pygame.display.flip()

pygame.quit()