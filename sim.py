import numpy as np
from numba import njit

@njit(cache=True)
def collide_bruteforce(pos, vel, radius):
    """Her ajan çifti kontrol edilir: O(N^2). Referans (yavaş) sürüm."""
    n = pos.shape[0]
    min_d = 2.0 * radius
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            d2 = dx * dx + dy * dy
            if d2 < min_d * min_d and d2 > 0.0:
                d = np.sqrt(d2)
                nx = dx / d
                ny = dy / d
                overlap = (min_d - d) * 0.5
                pos[i, 0] -= nx * overlap
                pos[i, 1] -= ny * overlap
                pos[j, 0] += nx * overlap
                pos[j, 1] += ny * overlap
                rel = (vel[j, 0] - vel[i, 0]) * nx + (vel[j, 1] - vel[i, 1]) * ny
                if rel < 0.0:
                    vel[i, 0] += rel * nx
                    vel[i, 1] += rel * ny
                    vel[j, 0] -= rel * nx
                    vel[j, 1] -= rel * ny
                count += 1
    return count

@njit(cache=True)
def _resolve(pos, vel, i, j, min_d):
    dx = pos[j, 0] - pos[i, 0]
    dy = pos[j, 1] - pos[i, 1]
    d2 = dx * dx + dy * dy
    if d2 < min_d * min_d and d2 > 0.0:
        d = np.sqrt(d2)
        nx = dx / d
        ny = dy / d
        overlap = (min_d - d) * 0.5
        pos[i, 0] -= nx * overlap
        pos[i, 1] -= ny * overlap
        pos[j, 0] += nx * overlap
        pos[j, 1] += ny * overlap
        rel = (vel[j, 0] - vel[i, 0]) * nx + (vel[j, 1] - vel[i, 1]) * ny
        if rel < 0.0:
            vel[i, 0] += rel * nx
            vel[i, 1] += rel * ny
            vel[j, 0] -= rel * nx
            vel[j, 1] -= rel * ny
        return 1
    return 0

@njit(cache=True)
def collide_grid(pos, vel, radius, w, h):
    """Spatial hashing: sadece komşu hücreler kontrol edilir (~O(N))."""
    n = pos.shape[0]
    min_d = 2.0 * radius
    cell = min_d
    gx = int(w // cell) + 1
    gy = int(h // cell) + 1

    # 1) Her ajanın hücresini bul, hücre başına sayım yap
    start = np.zeros(gx * gy + 1, np.int32)
    cell_of = np.empty(n, np.int32)
    for i in range(n):
        cx = min(max(int(pos[i, 0] // cell), 0), gx - 1)
        cy = min(max(int(pos[i, 1] // cell), 0), gy - 1)
        c = cy * gx + cx
        cell_of[i] = c
        start[c + 1] += 1
    for c in range(gx * gy):
        start[c + 1] += start[c]

    # 2) Ajanları hücre sırasına diz (counting sort)
    order = np.empty(n, np.int32)
    fill = start[:-1].copy()
    for i in range(n):
        c = cell_of[i]
        order[fill[c]] = i
        fill[c] += 1

    # 3) Her ajan için 3x3 komşu hücreyi tara
    count = 0
    for i in range(n):
        c = cell_of[i]
        cx = c % gx
        cy = c // gx
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                x2 = cx + ox
                y2 = cy + oy
                if x2 < 0 or x2 >= gx or y2 < 0 or y2 >= gy:
                    continue
                c2 = y2 * gx + x2
                for k in range(start[c2], start[c2 + 1]):
                    j = order[k]
                    if j > i:
                        count += _resolve(pos, vel, i, j, min_d)
    return count