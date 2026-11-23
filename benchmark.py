import platform
import time

import numpy as np

from sim import collide_bruteforce, collide_grid

W, H = 800, 600
RADIUS = 3.0
DT = 1 / 60
SIZES = [1000, 2000, 5000, 10000, 20000]
BUDGET_S = 3.0   # her ölçüm için yaklaşık süre
MIN_STEPS = 20
MAX_STEPS = 500


def make_agents(n):
    rng = np.random.default_rng(42)
    pos = rng.uniform([0, 0], [W, H], (n, 2))
    angle = rng.uniform(0, 2 * np.pi, n)
    vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * 60
    return pos, vel


def step(pos, vel, mode):
    pos += vel * DT
    hit_x = (pos[:, 0] < 0) | (pos[:, 0] > W)
    hit_y = (pos[:, 1] < 0) | (pos[:, 1] > H)
    vel[hit_x, 0] *= -1
    vel[hit_y, 1] *= -1
    pos[:, 0] = np.clip(pos[:, 0], 0, W)
    pos[:, 1] = np.clip(pos[:, 1], 0, H)
    if mode == "grid":
        collide_grid(pos, vel, RADIUS, W, H)
    else:
        collide_bruteforce(pos, vel, RADIUS)


def bench(n, mode):
    pos, vel = make_agents(n)
    for _ in range(5):  # ısınma (Numba derlemesi + cache)
        step(pos, vel, mode)
    times = []
    start = time.perf_counter()
    while len(times) < MIN_STEPS or (
        time.perf_counter() - start < BUDGET_S and len(times) < MAX_STEPS
    ):
        t0 = time.perf_counter()
        step(pos, vel, mode)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def fmt(x):
    return f"{x:.2f}".replace(".", ",")


def main():
    rows = []
    for n in SIZES:
        print(f"{n} ajan ölçülüyor...", flush=True)
        brute = bench(n, "brute")
        grid = bench(n, "grid")
        rows.append((n, brute, grid))

    lines = [
        "| Ajan | Brute-force O(N²) (ms) | Spatial hash grid (ms) | Hızlanma |",
        "|---|---|---|---|",
    ]
    for n, b, g in rows:
            lines.append(f"| {n:,} | {fmt(b)} | {fmt(g)} | {b / g:.1f}x |")
    table = "\n".join(lines)

    info = (
        f"Alan {W}x{H}, yarıçap {RADIUS} px, medyan adım süresi. "
        f"Python {platform.python_version()}, {platform.processor() or platform.machine()}."
    )
    print("\n" + table + "\n\n" + info)

    with open("benchmark_results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n\n" + info + "\n")
    print("\nbenchmark_results.md dosyasına yazıldı.")


if __name__ == "__main__":
    main()