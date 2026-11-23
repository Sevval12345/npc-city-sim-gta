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