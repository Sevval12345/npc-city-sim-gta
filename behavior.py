import numpy as np
from numba import njit

WANDER, FLEE, TO_COVER, HIDE = 0, 1, 2, 3

@njit(cache=True)
def seg_hits_rect(x0, y0, x1, y1, rx1, ry1, rx2, ry2):
    """Doğru parçası dikdörtgeni kesiyor mu? (slab yöntemi)"""
    dx = x1 - x0
    dy = y1 - y0
    tmin = 0.0
    tmax = 1.0
    if abs(dx) < 1e-9:
        if x0 < rx1 or x0 > rx2:
            return False
    else:
        t1 = (rx1 - x0) / dx
        t2 = (rx2 - x0) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    if abs(dy) < 1e-9:
        if y0 < ry1 or y0 > ry2:
            return False
    else:
        t1 = (ry1 - y0) / dy
        t2 = (ry2 - y0) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return True

@njit(cache=True)
def compute_cover_points(rects, sx, sy, radius, w, h, cover_pts, cover_ok):
    """Her bina için silaha göre arka taraftaki siper noktasını hesaplar."""
    for b in range(rects.shape[0]):
        cx = (rects[b, 0] + rects[b, 2]) * 0.5
        cy = (rects[b, 1] + rects[b, 3]) * 0.5
        hw = (rects[b, 2] - rects[b, 0]) * 0.5
        hh = (rects[b, 3] - rects[b, 1]) * 0.5
        if sx > rects[b, 0] and sx < rects[b, 2] and sy > rects[b, 1] and sy < rects[b, 3]:
            cover_ok[b] = False
            continue
        ux = cx - sx
        uy = cy - sy
        d = np.sqrt(ux * ux + uy * uy)
        if d < 1e-6:
            cover_ok[b] = False
            continue
        ux /= d
        uy /= d
        t = 1e9
        if abs(ux) > 1e-9:
            t = min(t, hw / abs(ux))
        if abs(uy) > 1e-9:
            t = min(t, hh / abs(uy))
        px = cx + ux * (t + radius + 8.0)
        py = cy + uy * (t + radius + 8.0)
        cover_pts[b, 0] = px
        cover_pts[b, 1] = py
        cover_ok[b] = px > 5.0 and px < w - 5.0 and py > 5.0 and py < h - 5.0

@njit(cache=True)
def apply_shot_goap(pos, vel, state, timer, target, rects, cover_pts, cover_ok,
                    sx, sy, hear_r, speed, flee_time, flee_cost, radius):
    """Silahı duyan her ajan en ucuz geçerli planı seçer: siper veya kaçış."""
    nb = rects.shape[0]
    heard = 0
    covered = 0
    for i in range(pos.shape[0]):
        dx = pos[i, 0] - sx
        dy = pos[i, 1] - sy
        d2 = dx * dx + dy * dy
        if d2 >= hear_r * hear_r:
            continue
        heard += 1

        best_b = -1
        best_cost = flee_cost
        for b in range(nb):
            if not cover_ok[b]:
                continue
            px = cover_pts[b, 0]
            py = cover_pts[b, 1]
            ddx = px - pos[i, 0]
            ddy = py - pos[i, 1]
            cost = np.sqrt(ddx * ddx + ddx * 0.0 + ddy * ddy) / speed
            if cost >= best_cost:
                continue
            blocked = False
            for r in range(nb):
                if seg_hits_rect(pos[i, 0], pos[i, 1], px, py,
                                 rects[r, 0] - radius, rects[r, 1] - radius,
                                 rects[r, 2] + radius, rects[r, 3] + radius):
                    blocked = True
                    break
            if not blocked:
                best_b = b
                best_cost = cost

        if best_b >= 0:
            px = cover_pts[best_b, 0]
            py = cover_pts[best_b, 1]
            ddx = px - pos[i, 0]
            ddy = py - pos[i, 1]
            d = np.sqrt(ddx * ddx + ddy * ddy)
            if d < 1e-6:
                d = 1.0
            vel[i, 0] = ddx / d * speed
            vel[i, 1] = ddy / d * speed
            target[i, 0] = px
            target[i, 1] = py
            state[i] = TO_COVER
            timer[i] = 5.0  # yolda en fazla 5 sn
            covered += 1
        else:
            d = np.sqrt(d2)
            if d < 1e-6:
                dx = 1.0
                dy = 0.0
                d = 1.0
            vel[i, 0] = dx / d * speed
            vel[i, 1] = dy / d * speed
            state[i] = FLEE
            timer[i] = flee_time
    return heard, covered

@njit(cache=True)
def _start_wander(vel, i, speed):
    a = np.random.uniform(0.0, 2.0 * np.pi)
    vel[i, 0] = np.cos(a) * speed
    vel[i, 1] = np.sin(a) * speed

@njit(cache=True)
def update_states(pos, vel, state, timer, target, dt, wander_speed, run_speed, hide_time):
    for i in range(vel.shape[0]):
        s = state[i]
        if s == FLEE:
            timer[i] -= dt
            if timer[i] <= 0.0:
                state[i] = WANDER
                _start_wander(vel, i, wander_speed)
        elif s == TO_COVER:
            dx = target[i, 0] - pos[i, 0]
            dy = target[i, 1] - pos[i, 1]
            d = np.sqrt(dx * dx + dy * dy)
            timer[i] -= dt
            if d < 4.0:
                state[i] = HIDE
                timer[i] = hide_time
                vel[i, 0] = 0.0
                vel[i, 1] = 0.0
            elif timer[i] <= 0.0:
                state[i] = WANDER
                _start_wander(vel, i, wander_speed)
            else:
                vel[i, 0] = dx / d * run_speed
                vel[i, 1] = dy / d * run_speed
        elif s == HIDE:
            vel[i, 0] = 0.0
            vel[i, 1] = 0.0
            timer[i] -= dt
            if timer[i] <= 0.0:
                state[i] = WANDER
                _start_wander(vel, i, wander_speed)

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