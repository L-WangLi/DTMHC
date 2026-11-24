"""Fixed-origin historical means, population standard deviations and log counts."""

import numpy as np


def history_tokens(trajs, index, window, seg=10, hmax=56, attr="x"):
    """Use only observations preceding each recent window; right-align valid segments.

    Return hx [N,H,2D+1], hpad [N,H] (True=padding), and hoff [N,H].
    Offsets are observation-index distances from segment endpoints to prediction.
    """
    d = getattr(trajs[0], attr).shape[1]
    hx = np.zeros((len(index), hmax, 2 * d + 1), dtype=np.float32)
    hpad = np.ones((len(index), hmax), dtype=bool)
    hoff = np.zeros((len(index), hmax), dtype=np.float32)
    cache = {}
    for n, (k, e) in enumerate(index):
        s = e - window + 1
        if s <= 0:
            continue
        if k not in cache:
            x = getattr(trajs[k], attr).astype(np.float64)
            z = np.zeros((1, d))
            cache[k] = (np.vstack([z, np.cumsum(x, 0)]),
                        np.vstack([z, np.cumsum(x ** 2, 0)]))
        c1, c2 = cache[k]
        starts = np.arange(0, s, seg)
        ends = np.minimum(starts + seg, s)
        cnt = (ends - starts)[:, None].astype(np.float64)
        mean = (c1[ends] - c1[starts]) / cnt
        std = np.sqrt(np.maximum((c2[ends] - c2[starts]) / cnt - mean ** 2, 0.0))
        m = len(starts)
        if m > hmax:
            raise ValueError(f"{m} history segments exceed configured capacity {hmax}")
        sl = slice(hmax - m, hmax)
        hx[n, sl, :-1] = np.concatenate([mean, std], 1)
        hx[n, sl, -1] = np.log(cnt[:, 0])
        hpad[n, sl] = False
        hoff[n, sl] = e - (ends - 1)
    return hx, hpad, hoff
