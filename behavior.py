import numpy as np
from numba import njit

@njit(cache=True)
def apply_shot(pos, vel, state, timer, sx, sy, hear_r, flee_speed, flee_time):
    """Silah sesi: duyma yarıçapındaki ajanlar sesten uzağa kaçmaya başlar."""
    heard = 0
    for i in range(pos.shape[0]):
        dx = pos[i, 0] - sx
        dy = pos[i, 1] - sy
        d2 = dx * dx + dy * dy
        if d2 < hear_r * hear_r:
            d = np.sqrt(d2)
            if d < 1e-6:
                dx = 1.0
                dy = 0.0
                d = 1.0
            vel[i, 0] = dx / d * flee_speed
            vel[i, 1] = dy / d * flee_speed
            state[i] = 1
            timer[i] = flee_time
            heard += 1
    return heard

@njit(cache=True)
def update_states(vel, state, timer, dt, wander_speed):
    """Kaçış süresi bitince ajan sakinleşip rastgele yöne gezmeye döner."""
    for i in range(vel.shape[0]):
        if state[i] == 1:
            timer[i] -= dt
            if timer[i] <= 0.0:
                state[i] = 0
                a = np.random.uniform(0.0, 2.0 * np.pi)
                vel[i, 0] = np.cos(a) * wander_speed
                vel[i, 1] = np.sin(a) * wander_speed


@njit(cache=True)
def collide_rects(pos, vel, rects, radius):
    """Dikdörtgen binalara girmeyi engeller, ajanı en yakın kenara iter."""
    for i in range(pos.shape[0]):
        px = pos[i, 0]
        py = pos[i, 1]
        for r in range(rects.shape[0]):
            x1 = rects[r, 0] - radius
            y1 = rects[r, 1] - radius
            x2 = rects[r, 2] + radius
            y2 = rects[r, 3] + radius
            if px > x1 and px < x2 and py > y1 and py < y2:
                dl = px - x1
                dr = x2 - px
                dtop = py - y1
                dbot = y2 - py
                m = min(min(dl, dr), min(dtop, dbot))
                if m == dl:
                    pos[i, 0] = x1
                    vel[i, 0] = -abs(vel[i, 0])
                elif m == dr:
                    pos[i, 0] = x2
                    vel[i, 0] = abs(vel[i, 0])
                elif m == dtop:
                    pos[i, 1] = y1
                    vel[i, 1] = -abs(vel[i, 1])
                else:
                    pos[i, 1] = y2
                    vel[i, 1] = abs(vel[i, 1])