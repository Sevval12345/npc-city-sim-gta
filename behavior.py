import numpy as np
from numba import njit

WANDER, FLEE, TO_COVER, HIDE = 0, 1, 2, 3
INF = 1e9
SQRT2 = 1.41421356

# ---------------------------------------------------------------- ızgara
@njit(cache=True)
def build_blocked(rects, cell, blocked):
    """Bir binayla kesişen her hücre engelli."""
    gh, gw = blocked.shape
    for gy in range(gh):
        for gx in range(gw):
            x1 = gx * cell
            x2 = x1 + cell
            y1 = gy * cell
            y2 = y1 + cell
            b = False
            for r in range(rects.shape[0]):
                if x1 < rects[r, 2] and x2 > rects[r, 0] and y1 < rects[r, 3] and y2 > rects[r, 1]:
                    b = True
                    break
            blocked[gy, gx] = b


@njit(cache=True)
def _heap_push(hk, hv, size, key, val):
    i = size
    hk[i] = key
    hv[i] = val
    while i > 0:
        p = (i - 1) // 2
        if hk[p] <= hk[i]:
            break
        tk = hk[p]
        hk[p] = hk[i]
        hk[i] = tk
        tv = hv[p]
        hv[p] = hv[i]
        hv[i] = tv
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop(hk, hv, size):
    """En küçük eleman hk[0]/hv[0]'dır; çağırmadan önce oku. Yeni boyutu döner."""
    size -= 1
    if size > 0:
        hk[0] = hk[size]
        hv[0] = hv[size]
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            s = i
            if l < size and hk[l] < hk[s]:
                s = l
            if r < size and hk[r] < hk[s]:
                s = r
            if s == i:
                break
            tk = hk[s]
            hk[s] = hk[i]
            hk[i] = tk
            tv = hv[s]
            hv[s] = hv[i]
            hv[i] = tv
            i = s
    return size


@njit(cache=True)
def dijkstra_field(blocked, gx0, gy0, cell, dist):
    """(gx0, gy0) hedef hücresinden her hücreye en kısa yol uzunluğu (piksel)."""
    gh, gw = blocked.shape
    for y in range(gh):
        for x in range(gw):
            dist[y, x] = INF
    cap = gw * gh * 8 + 16
    hk = np.empty(cap, np.float64)
    hv = np.empty(cap, np.int32)
    size = 0
    dist[gy0, gx0] = 0.0
    size = _heap_push(hk, hv, size, 0.0, gy0 * gw + gx0)
    while size > 0:
        k = hk[0]
        v = hv[0]
        size = _heap_pop(hk, hv, size)
        cx = v % gw
        cy = v // gw
        if k > dist[cy, cx]:
            continue
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                    continue
                if blocked[ny, nx]:
                    continue
                step = 1.0
                if dx != 0 and dy != 0:
                    if blocked[cy, nx] or blocked[ny, cx]:
                        continue  # köşe kesme yok
                    step = SQRT2
                nd = k + step * cell
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    size = _heap_push(hk, hv, size, nd, ny * gw + nx)


@njit(cache=True)
def compute_cover_points(rects, sx, sy, w, h, offset, cover_pts, cover_ok):
    """Her bina için silaha göre arka taraftaki siper noktası."""
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
        px = cx + ux * (t + offset)
        py = cy + uy * (t + offset)
        cover_pts[b, 0] = px
        cover_pts[b, 1] = py
        cover_ok[b] = px > 5.0 and px < w - 5.0 and py > 5.0 and py < h - 5.0


@njit(cache=True)
def compute_fields(blocked, cover_pts, cover_ok, dist, cell):
    """Her geçerli siper noktası için mesafe alanı hesaplar."""
    gh, gw = blocked.shape
    for b in range(cover_pts.shape[0]):
        gx = int(cover_pts[b, 0] // cell)
        gy = int(cover_pts[b, 1] // cell)
        ok = cover_ok[b] and gx >= 0 and gx < gw and gy >= 0 and gy < gh
        if ok and blocked[gy, gx]:
            ok = False
        cover_ok[b] = ok
        if ok:
            dijkstra_field(blocked, gx, gy, cell, dist[b])
        else:
            for y in range(gh):
                for x in range(gw):
                    dist[b, y, x] = INF


@njit(cache=True)
def _local_dist(field, cx, cy, cell):
    """Ajanın hücresindeki değer; engelli sliver hücredeyse komşulardan tahmin."""
    gh, gw = field.shape
    best = field[cy, cx]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                continue
            step = 1.0
            if dx != 0 and dy != 0:
                step = SQRT2
            v = field[ny, nx] + step * cell
            if v < best:
                best = v
    return best


@njit(cache=True)
def _best_neighbor(field, blocked, cx, cy, cell):
    """Hedefe en kısa yolu veren geçerli komşu hücre (yoksa -1, -1)."""
    gh, gw = blocked.shape
    best = 1e18
    bx = -1
    by = -1
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                continue
            if blocked[ny, nx] or field[ny, nx] >= 1e8:
                continue
            step = 1.0
            if dx != 0 and dy != 0:
                if blocked[cy, nx] or blocked[ny, cx]:
                    continue
                step = SQRT2
            v = field[ny, nx] + step * cell
            if v < best:
                best = v
                bx = nx
                by = ny
    return bx, by


# ------------------------------------------------------------- davranış
@njit(cache=True)
def apply_shot_goap(pos, vel, state, timer, goal, dist, blocked, cover_pts, cover_ok,
                    sx, sy, hear_r, speed, flee_time, flee_cost, cell):
    """Silahı duyan her ajan en ucuz planı seçer: yol uzunluğuna göre siper veya kaçış."""
    gh, gw = blocked.shape
    nb = cover_pts.shape[0]
    heard = 0
    covered = 0
    for i in range(pos.shape[0]):
        dx = pos[i, 0] - sx
        dy = pos[i, 1] - sy
        d2 = dx * dx + dy * dy
        if d2 >= hear_r * hear_r:
            continue
        heard += 1

        cx = min(max(int(pos[i, 0] // cell), 0), gw - 1)
        cy = min(max(int(pos[i, 1] // cell), 0), gh - 1)
        best_b = -1
        best_cost = flee_cost
        for b in range(nb):
            if not cover_ok[b]:
                continue
            d = _local_dist(dist[b], cx, cy, cell)
            if d >= 1e8:
                continue
            cost = d / speed
            if cost < best_cost:
                best_cost = cost
                best_b = b

        if best_b >= 0:
            state[i] = TO_COVER
            goal[i] = best_b
            timer[i] = 8.0  # yolda en fazla 8 sn
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
def update_states(pos, vel, state, timer, goal, dist, blocked, cover_pts, cover_ok,
                  cell, dt, wander_speed, run_speed, hide_time):
    gh, gw = blocked.shape
    for i in range(vel.shape[0]):
        s = state[i]
        if s == FLEE:
            timer[i] -= dt
            if timer[i] <= 0.0:
                state[i] = WANDER
                _start_wander(vel, i, wander_speed)
        elif s == TO_COVER:
            timer[i] -= dt
            b = goal[i]
            if timer[i] <= 0.0 or not cover_ok[b]:
                state[i] = WANDER
                _start_wander(vel, i, wander_speed)
                continue
            px = cover_pts[b, 0]
            py = cover_pts[b, 1]
            dx = px - pos[i, 0]
            dy = py - pos[i, 1]
            if np.sqrt(dx * dx + dy * dy) < 4.0:
                state[i] = HIDE
                timer[i] = hide_time
                vel[i, 0] = 0.0
                vel[i, 1] = 0.0
                continue
            cx = min(max(int(pos[i, 0] // cell), 0), gw - 1)
            cy = min(max(int(pos[i, 1] // cell), 0), gh - 1)
            gcx = int(px // cell)
            gcy = int(py // cell)
            tx = px
            ty = py
            if not (cx == gcx and cy == gcy):
                nx, ny = _best_neighbor(dist[b], blocked, cx, cy, cell)
                if nx < 0:
                    state[i] = WANDER
                    _start_wander(vel, i, wander_speed)
                    continue
                if not (nx == gcx and ny == gcy):
                    tx = (nx + 0.5) * cell
                    ty = (ny + 0.5) * cell
            ddx = tx - pos[i, 0]
            ddy = ty - pos[i, 1]
            dd = np.sqrt(ddx * ddx + ddy * ddy)
            if dd < 1e-6:
                dd = 1.0
            vel[i, 0] = ddx / dd * run_speed
            vel[i, 1] = ddy / dd * run_speed
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