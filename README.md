# NPC City Sim

Python ile sıfırdan yazılmış 2D kalabalık simülasyonu. NumPy + Numba kullanır, oyun motoru yoktur.

## Çarpışma benchmark'ı

Simülasyon adımı süresi (ms, medyan), 800x600 alan, yarıçap 3 px. Düşük olan iyi.
Ölçüm `benchmark.py` ile otomatik yapılır (çizim dahil değil).

| Ajan | Brute-force O(N²) (ms) | Spatial hash grid (ms) | Hızlanma |
|---|---|---|---|
| 1.000 | 0,64 | 0,13 | 4,9x |
| 2.000 | 2,24 | 0,22 | 10,1x |
| 5.000 | 13,73 | 0,80 | 17,1x |
| 10.000 | 54,16 | 1,77 | 30,6x |
| 20.000 | 221,13 | 6,20 | 35,7x |

Brute-force, ajan sayısı ikiye katlanınca yaklaşık 4 kat yavaşlıyor (O(N²)).
Grid çok daha yavaş büyüyor. En yoğun durumda (20.000 ajan) hücre başına düşen
ajan arttığı için süre doğrusaldan biraz hızlı artıyor, ama fark yine de açılıyor.

Test makinesi: AMD işlemci, Python 3.13, Windows.

## Çalıştırma

    pip install -r requirements.txt
    python main.py

SPACE tuşu brute-force ve grid modları arasında geçiş yapar.
Benchmark'ı yeniden üretmek için: `python benchmark.py`

SPACE tuşu brute-force ve grid modları arasında geçiş yapar.