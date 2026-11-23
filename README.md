# NPC City Sim

Python ile sıfırdan yazılmış 2D kalabalık simülasyonu. NumPy + Numba kullanır, oyun motoru yoktur.

## Çarpışma benchmark'ı

Simülasyon adımı süresi (ms), 800x600 alan, yarıçap 3 px. Düşük olan iyi.

| Ajan | Brute-force O(N²) | Spatial hash grid | Hızlanma |
|---|---|---|---|
| 1.000 | 0,65 | 0,18 | 3,7x |
| 2.000 | 2,25 | 0,32 | 7x |
| 5.000 | 14 | 0,70 | 20x |
| 10.000 | 58 | 1,5 | 39x |
| 20.000 | 219 | 6,5 | 34x |

Not: Ölçümler pencere başlığından elle okunmuş yaklaşık değerlerdir. Çok yüksek yoğunlukta (20.000 ajan) hızlanma daralıyor.

## Çalıştırma
    pip install -r requirements.txt
    python main.py

SPACE tuşu brute-force ve grid modları arasında geçiş yapar.