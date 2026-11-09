"""Lilium orientalis - procedural growing 3D lily (NumPy + PyVista).

    python lilium_orientalis.py              animation (about 16 s), fixed camera
    python lilium_orientalis.py --t 0.5      still frame at time t (0..1)
    python lilium_orientalis.py --shot a.png --t 0.5   save a frame, no window
    python lilium_orientalis.py --check      self-check of the geometry, no window

Stages (global time t from 0 to 1, 1 s = 0.0625):
  0.06-0.38 green wireframe stems rise and fill, branches fork off, leaves shoot out as lines and widen
  0.26-0.49 stamens + pistil appear first (exposed flower centres)
  0.37-0.80 tepal wireframes (with dots) unfold from small to open, then fill white/pink, one by one
  0.70-0.90 dark papillae, the two closed buds fill in      0.90-1.00 finished plant
  particles: light up around the plant during 0.00-0.15; pollen grains rise from the anthers during 0.34-1.00
"""
import sys
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- data container
@dataclass
class BotanicalElement:
    kind: str                  # "stem", "leaf", "tepal", "papillae", "stamen", "pistil"
    P: np.ndarray              # vertex positions (nu, nv, 3) float32
    C: np.ndarray              # vertex colors RGB 0..1 (nu, nv, 3) float32
    wire_t0: float = 0.0       # wireframe fades in between wire_t0 and wire_t1
    wire_t1: float = 0.0
    surf_t0: float = 0.0       # filled surface fades in between surf_t0 and surf_t1
    surf_t1: float = 0.0
    material: dict = field(default_factory=dict)   # wire_stride, wire_color, dots, points_only

    @property
    def u_grid(self):
        return self.P.shape[0]

    @property
    def v_grid(self):
        return self.P.shape[1]


def element(kind, P, C, wire, surf, **material):
    f = lambda a: np.asarray(a, dtype=np.float32)
    return BotanicalElement(kind, f(P), f(C), wire[0], wire[1], min(surf[0], 1.0), min(surf[1], 1.0), material)


# ---------------------------------------------------------------- math helpers
def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def smoothstep(a, b, x):
    """0 below a, 1 above b, smooth S-curve in between."""
    t = np.clip((np.asarray(x, dtype=float) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    """Mix colors: (1-t)*a + t*b. t may be an array of any shape; the color axis is added."""
    t = np.asarray(t, dtype=float)[..., None]
    return (1 - t) * np.asarray(a) + t * np.asarray(b)


def bezier(ctrl, t):
    """Cubic Bezier: weighted average of 4 control points."""
    t = np.asarray(t)[:, None]
    p0, p1, p2, p3 = ctrl
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def bishop_frames(T):
    """Twist-free normals along a curve: rotate the previous normal only as much as the tangent turned."""
    N = np.zeros_like(T)
    helper = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    N[0] = normalize(np.cross(T[0], np.cross(helper, T[0])))
    for i in range(1, len(T)):
        axis = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(axis)
        if s < 1e-9:
            N[i] = N[i - 1]
            continue
        axis /= s
        ang = np.arctan2(s, np.dot(T[i - 1], T[i]))
        N[i] = (N[i - 1] * np.cos(ang) + np.cross(axis, N[i - 1]) * np.sin(ang)
                + axis * np.dot(axis, N[i - 1]) * (1 - np.cos(ang)))
    return T, N, np.cross(T, N)


def make_tube(center, radius, nv=8):
    """Tube around a polyline. radius: (nu,) or (nu, nv). Returns points (nu, nv, 3) and frames."""
    T, N, B = bishop_frames(normalize(np.gradient(center, axis=0)))
    theta = np.linspace(0, 2 * np.pi, nv)           # last point == first, closes the tube
    radius = np.broadcast_to(radius if np.ndim(radius) == 2 else np.asarray(radius)[:, None], (len(center), nv))
    ring = np.cos(theta)[None, :, None] * N[:, None, :] + np.sin(theta)[None, :, None] * B[:, None, :]
    return center[:, None, :] + radius[..., None] * ring, (T, N, B)


# ---------------------------------------------------------------- palette
GREEN_LOW, GREEN_HIGH = (0.06, 0.38, 0.08), (0.55, 0.85, 0.30)
WHITE, PINK = (1.0, 0.97, 0.97), (0.93, 0.30, 0.52)
ANTHER, STIGMA = (0.80, 0.38, 0.12), (0.80, 0.35, 0.70)
WIRE_GREEN, WIRE_PINK, WIRE_WHITE = (0.65, 0.95, 0.55), (1.0, 0.82, 0.90), (0.95, 0.95, 0.85)


# ---------------------------------------------------------------- stem, buds and leaves
def rising(P, t0, dur):
    """Shape at time t: rows above the growing tip collapse onto the tip (it rises, it is not scaled)."""
    nu = len(P)
    def at(t):
        x = np.minimum(np.arange(nu), float(np.clip((t - t0) / dur, 0, 1)) * (nu - 1))
        lo = np.floor(x).astype(int)
        f = (x - lo)[:, None, None]
        return (1 - f) * P[lo] + f * P[np.minimum(lo + 1, nu - 1)]
    return at


def stem_ctrl(base, top, wobble=0.03):
    """Almost straight Bezier: points 1 and 2 lie on the line, nudged sideways a little."""
    base, top = np.asarray(base, float), np.asarray(top, float)
    return np.array([base, base + (top - base) * 0.33 + [wobble, -wobble, 0],
                     base + (top - base) * 0.66 + [-wobble, wobble, 0], top])


def make_stem(ctrl, t0, dur, r0=0.026, nu=40, nv=8):
    """Green wireframe rises first, then fills in. Returns element, frames, centerline."""
    u = np.linspace(0, 1, nu)
    center = bezier(ctrl, u)
    P, frames = make_tube(center, r0 * (1 - 0.4 * u), nv)
    C = np.broadcast_to(lerp(GREEN_LOW, GREEN_HIGH, smoothstep(0.3, 1, u))[:, None, :], P.shape)
    return element("stem", P, C, (t0, t0 + 0.03), (t0 + 0.5 * dur, t0 + dur + 0.08), wire_stride=(8, 2),
                   wire_color=WIRE_GREEN, wire_keep=0.4, rebuild=rising(P, t0, dur), rebuild_win=(t0, t0 + dur)), frames, center


def make_bud(ctrl, t0, dur, nu=30, nv=13):
    """Closed lily bud on a thin stalk: spindle-shaped tube, green with a pink tip. Fills in late (0.76-0.88)."""
    u = np.linspace(0, 1, nu)
    s = np.clip((u - 0.45) / 0.55, 0, 1)                                      # 0 on the stalk, 0..1 along the bud
    P, _ = make_tube(bezier(ctrl, u), 0.012 + 0.05 * np.sin(np.pi * s) ** 1.3 * (1 - 0.4 * s), nv)
    col = lerp(lerp(GREEN_LOW, GREEN_HIGH, smoothstep(0.3, 0.8, u)), PINK, 0.6 * smoothstep(0.75, 1, u))
    return element("bud", P, np.broadcast_to(col[:, None, :], P.shape), (t0, t0 + 0.03), (0.76, 0.88),
                   wire_stride=(3, 3), wire_color=WIRE_GREEN, wire_keep=0.0, rebuild=rising(P, t0, dur),
                   rebuild_win=(t0, t0 + dur))


def lance_width(u):
    """Narrow base, widest near 30 percent, pointed tip."""
    w = u ** 0.5 * (1 - u) ** 0.9
    return w / w.max()


def make_leaf(base, T, side_hint, length, max_width, t0, pitch0=0.35, droop=1.2, nu=28, nv=9):
    """From t0: a thin green line shoots out (0.06), then widens into the blade (0.06), then fills in."""
    u = np.linspace(0, 1, nu)
    d = normalize(side_hint - np.dot(side_hint, T) * T)            # outward, horizontal-ish
    phi = pitch0 - droop * u ** 2                                   # angle above horizontal; tip droops most
    step = (np.cos(phi)[:, None] * d + np.sin(phi)[:, None] * T) * (length / (nu - 1))
    center = base + np.vstack([np.zeros(3), np.cumsum(step[1:], axis=0)])
    side = normalize(np.cross(T, d))                                # across the blade
    normal = normalize(np.cross(normalize(step), side))             # up from the blade
    v = np.linspace(-1, 1, nv)
    w = lance_width(u) * max_width
    fold = 0.35 * w[:, None] * np.abs(v)[None, :]                   # V-fold along the midrib
    P = (center[:, None, :] + (w[:, None] * v[None, :])[..., None] * side[None, None, :]
         + fold[..., None] * normal[:, None, :])
    shade = np.clip(0.6 * (1 - np.abs(v)[None, :]) ** 2 + 0.1 * np.sin(u * 14)[:, None], 0, 1)
    C = lerp((0.04, 0.26, 0.07), (0.24, 0.62, 0.16), shade)
    line, blade = rising(np.repeat(center[:, None, :], nv, axis=1), t0, 0.06), rising(P, t0, 0.06)
    at = lambda t: line(t) + (blade(t) - line(t)) * smoothstep(t0 + 0.06, t0 + 0.12, t)
    return element("leaf", P, C, (t0, t0 + 0.08), (t0 + 0.10, t0 + 0.16), wire_stride=(5, 4),
                   wire_color=WIRE_GREEN, wire_keep=0.6, rebuild=at, rebuild_win=(t0, t0 + 0.12))


def make_leaves(frames, center, count, phase, scale, t0, dur, top=0.40):
    """count leaves on the lower part (up to `top`) of a stem that rises during t0 .. t0+dur."""
    T, N, B = frames
    leaves = []
    for k in range(count):
        f = k / max(count - 1, 1)
        i = int((0.04 + (top - 0.04) * f) * (len(center) - 1))
        a = phase + k * np.deg2rad(137.5)                           # golden angle
        hint = np.cos(a) * N[i] + np.sin(a) * B[i]
        size = scale * (1 - 0.45 * f)                               # upper leaves smaller
        leaves.append(make_leaf(center[i], T[i], hint, 0.75 * size, 0.15 * size, t0=t0 + dur * i / (len(center) - 1)))
    return leaves


# ---------------------------------------------------------------- flower
def flower_frame(axis):
    Z = normalize(axis)
    X = normalize(np.cross([0, 1.0, 0], Z))
    return X, np.cross(Z, X), Z


def to_world(local, c, frame):
    X, Y, Z = frame
    return c + local[..., 0:1] * X + local[..., 1:2] * Y + local[..., 2:3] * Z


def spin(p, az):
    c, s = np.cos(az), np.sin(az)
    return np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1)


def opened(kw, o):
    """Tepal shape between a small closed bud (o=0: short, narrow, nearly upright) and the open star (o=1)."""
    bud = dict(a0=0.12, a1=0.25, cup=0.6, L=0.35 * kw["L"], W=0.25 * kw["W"])
    return {**kw, **{q: bud[q] + (kw[q] - bud[q]) * o for q in bud}}


def tepal_local(u, v, L, W, a0, a1, cup, phase, lift=0.0):
    """Tepal in local coords: x outward, y across, z up the flower axis.
    The midrib bends: its angle from the axis grows from a0 by a1 over the length
    (closed-form integral of sin/cos). Width: narrow base, wide belly, pointed tip.
    Edges curl up (cup) and ripple (wave)."""
    alpha = a0 + a1 * u
    x = L * (np.cos(a0) - np.cos(alpha)) / a1
    z = L * (np.sin(alpha) - np.sin(a0)) / a1
    y = v * W * (0.12 * (1 - u) + np.clip(np.sin(np.pi * u ** 0.8), 0, None) ** 0.9)
    h = cup * y ** 2 / W + 0.05 * L * np.sin(9 * u + phase) * np.abs(v) ** 3 + lift   # along the surface normal
    return np.stack([x - np.cos(alpha) * h, y, z + np.sin(alpha) * h], axis=-1)


def make_flower(c, axis, s, seed, fd=0.0):
    """Six tepals (two rings of 3), six stamens, one pistil. fd: delay of this whole flower.
    Order: stamens + pistil first -> tepal wireframes unfold from small to open -> fill white/pink -> papillae."""
    frame, rng, out = flower_frame(axis), np.random.default_rng(seed), []
    u, v = np.linspace(0, 1, 25), np.linspace(-1, 1, 13)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rings = [dict(L=0.80 * s, W=0.30 * s, a0=1.00, a1=1.10, cup=0.25, off=0.0),        # a0+a1 > pi/2: tips curl back
             dict(L=0.74 * s, W=0.34 * s, a0=0.90, a1=1.10, cup=0.30, off=np.pi / 3)]
    for r, ring in enumerate(rings):
        for k in range(3):
            n, az = 3 * r + k, ring["off"] + k * 2 * np.pi / 3
            kw = {q: ring[q] for q in ("L", "W", "a0", "a1", "cup")}
            d = fd + 0.008 * n                                                # every tepal starts a bit later
            w0, w1 = 0.38 + d, 0.60 + d                                       # the wireframe unfolds during w0 .. w1
            grid_at = lambda t, kw=kw, az=az, n=n, w0=w0, w1=w1: to_world(
                spin(tepal_local(uu, vv, phase=n, **opened(kw, smoothstep(w0, w1, t))), az), c, frame)
            P = grid_at(1.0)
            mix = np.clip(smoothstep(0.05, 0.75, uu + 0.35 * np.abs(vv) ** 1.5)   # pink core -> white edge
                          - 0.18 * np.cos(vv * 3 * np.pi) ** 2 * (1 - uu), 0, 1)   # pink rays
            stripe = np.exp(-(vv / 0.15) ** 2) * (1 - smoothstep(0.3, 0.85, uu))   # magenta midrib stripe
            col = lerp(lerp(PINK, WHITE, mix), (0.72, 0.08, 0.38), 0.8 * stripe)
            out.append(element("tepal", P, col, (w0, w0 + 0.12), (0.62 + d, 0.70 + d),
                               wire_stride=(2, 2), wire_color=WIRE_PINK, wire_keep=0.0,
                               dots=True, dots_win=(w0 - 0.01, w0 + 0.10), rebuild=grid_at, rebuild_win=(w0, w1)))
            pu = 0.05 + 0.45 * rng.random(70) ** 1.6                          # dark papillae, densest near the core
            pv_ = np.clip(rng.normal(0, 0.3, 70), -0.8, 0.8)
            pts_at = lambda t, kw=kw, az=az, n=n, pu=pu, pv_=pv_, w0=w0, w1=w1: to_world(
                spin(tepal_local(pu, pv_, phase=n, lift=0.006 * s, **opened(kw, smoothstep(w0, w1, t))), az),
                c, frame)[:, None, :]
            out.append(element("papillae", pts_at(1.0), np.full((70, 1, 3), (0.42, 0.04, 0.20)),
                               (0, 0), (0.70 + d, 0.80 + d), points_only=True, rebuild=pts_at, rebuild_win=(w0, w1)))
    u = np.linspace(0, 1, 22)
    for k in range(6):                                                         # stamens: curved filaments
        az = k * np.pi / 3 + 0.26
        ca, sa = np.cos(az), np.sin(az)
        flare, h_tip = (0.24 + 0.03 * (k % 2)) * s, (0.58 + 0.04 * (k % 3)) * s   # short: stays inside the bloom
        ctrl = np.array([[0.028 * ca, 0.028 * sa, 0.02], [0.07 * ca, 0.07 * sa, 0.22],
                         [0.17 * ca, 0.17 * sa, 0.45]]) * s
        ctrl = np.vstack([ctrl, [flare * ca, flare * sa, h_tip]])
        r = 0.009 * s * (1 - 0.5 * u) * (1 + 2.0 * smoothstep(0.80, 0.87, u)) * (1 - smoothstep(0.96, 1, u))
        P, _ = make_tube(to_world(bezier(ctrl, u), c, frame), r, 8)           # tapers, anther bulge at the tip
        col = lerp(lerp((0.55, 0.80, 0.45), WHITE, smoothstep(0.1, 0.6, u)), ANTHER, smoothstep(0.80, 0.87, u))
        d = fd + 0.01 * k                                                      # before the tepals: exposed flower centre
        out.append(element("stamen", P, np.broadcast_to(col[:, None, :], P.shape), (0.26 + d, 0.32 + d),
                           (0.30 + d, 0.38 + d), wire_stride=(4, 2), wire_color=WIRE_WHITE))
    H = 0.62 * s                                                               # pistil
    loc = np.stack([0.10 * s * u ** 2, 0.04 * s * u ** 2, H * u], axis=-1)
    theta = np.linspace(0, 2 * np.pi, 13)                                     # 12 unique points: 4 per lobe
    r = (0.016 * s * (1 - 0.3 * u) * (1 + 1.2 * smoothstep(0.84, 0.9, u)) * (1 - smoothstep(0.97, 1, u)))[:, None]
    r = r * (1 + 0.45 * np.cos(3 * theta))[None, :] ** (u > 0.84)[:, None]    # three lobes above u=0.84
    P, _ = make_tube(to_world(loc, c, frame), r, 13)
    col = lerp(GREEN_LOW, STIGMA, smoothstep(0.75, 0.9, u))
    out.append(element("pistil", P, np.broadcast_to(col[:, None, :], P.shape), (0.27 + fd, 0.33 + fd), (0.31 + fd, 0.39 + fd),
                       wire_stride=(4, 2), wire_color=WIRE_PINK))
    return out


# ---------------------------------------------------------------- plant
def build_plant():
    """One main stem (flower upper left); two branches fork off it (flowers middle right, lower left)
    and two thin stalks carry closed buds above the flowers."""
    main_t0, main_dur = 0.06, 0.22                       # main stem rises during 0.06 .. 0.28
    main_ctrl = stem_ctrl((0, 0, 0), (0, 0, 1.75), wobble=0.0)   # straight up
    elements = []
    #       fork at   top               leaves phase flower scale, delay   (fork=None: the main stem itself)
    specs = [(None, (0.00, 0.00, 1.75), 5, 0.0, 0.68, 0.00),
             (0.45, (0.38, 0.10, 1.45), 1, 2.0, 0.68, 0.03),
             (0.30, (-0.45, -0.10, 1.15), 1, 4.0, 0.58, 0.06)]
    main_center = bezier(main_ctrl, np.linspace(0, 1, 40))
    for fork, top, n_leaves, phase, fs, fd in specs:
        base = (0, 0, 0) if fork is None else main_center[int(fork * 39)]
        t0, dur = (main_t0, main_dur) if fork is None else (main_t0 + main_dur * fork, 0.10)
        stem, (T, N, B), center = make_stem(main_ctrl if fork is None else stem_ctrl(base, top), t0, dur, r0=0.026 if fork is None else 0.018)
        elements.append(stem)
        elements += make_leaves((T, N, B), center, n_leaves, phase, 0.8, t0, dur, top=0.55 if fork is None else 0.2)
        out = normalize(np.array([top[0] - base[0], top[1] - base[1], 0.0]))   # flower leans outward
        elements += make_flower(center[-1], T[-1] + 0.5 * out + [0, -0.9, 0], fs, seed=len(elements), fd=fd)
    for fork, top in ((0.70, (-0.90, 0.0, 2.10)), (0.70, (0.30, 0.70, 2.65))):  # two closed buds; positions clear the petals (see check)
        t0 = main_t0 + main_dur * fork
        elements.append(make_bud(stem_ctrl(main_center[int(fork * 39)], top, wobble=0.01), t0, 0.10))
    return elements


def make_pollen(elements, n_amb=60, n_grain=36, seed=3):
    """Glowing particles. Ambient ones gather around the flowers (plus a few strays) and light up one by one
    during t = 0 .. 0.15; pollen grains leave the anther tips once the stamens are out and float upward."""
    rng = np.random.default_rng(seed)
    centres = np.array([e.P[0].mean(axis=0) for e in elements if e.kind == "pistil"])
    anthers = np.array([e.P[19].mean(axis=0) for e in elements if e.kind == "stamen"])   # u ~ 0.9: anther bulge
    near = int(0.7 * n_amb)
    base = np.vstack([centres[rng.integers(0, len(centres), near)] + rng.normal(0, 0.45, (near, 3)),
                      rng.uniform([-0.9, -0.5, 0.2], [0.9, 0.5, 2.5], (n_amb - near, 3)),
                      anthers[rng.integers(0, len(anthers), n_grain)]])
    n = n_amb + n_grain
    palette = np.array([(1.0, 1.0, 1.0), (1.0, 0.92, 0.6), (1.0, 0.8, 0.9)])      # white, pale yellow, pale pink
    return dict(base=base, freq=rng.uniform(0.2, 0.7, (n, 3)), ph=rng.uniform(0, 2 * np.pi, (n, 3)),
                birth=np.concatenate([rng.uniform(0.0, 0.15, n_amb), rng.uniform(0.34, 0.9, n_grain)]),
                rise=np.r_[np.zeros(n_amb), np.ones(n_grain)], twinkle=rng.uniform(0, 2 * np.pi, n),
                rgb=palette[rng.choice(3, n, p=[0.5, 0.3, 0.2])])


def pollen_at(p, t, secs):
    """Positions (n, 3) and brightness 0..1 at animation time t and wall-clock secs (for the sway)."""
    age = np.clip(t - p["birth"], 0, None)
    pts = p["base"] + 0.07 * np.sin(p["freq"] * secs + p["ph"]) + (0.8 * p["rise"] * age)[:, None] * [0.15, 0, 1]
    alpha = smoothstep(0, 0.04, age) * (0.6 + 0.4 * np.sin(1.7 * secs + p["twinkle"]))   # fade in, then twinkle
    return pts.astype(np.float32), alpha


def pollen_rgba(p, alpha):
    return (np.column_stack([p["rgb"], alpha]) * 255).astype(np.uint8)


# ---------------------------------------------------------------- renderer (PyVista)
def flat(A):
    """(nu, nv, 3) -> (nu*nv, 3) with u running fastest: the point order of a StructuredGrid."""
    return A.transpose(1, 0, 2).reshape(-1, 3)


def isolines(nu, nv, su, sv):
    """Only every su-th u-line and sv-th v-line (plus the last one), cut into 2-point segments.
    Each segment gets a draw key in 0..1: lines start one after another (rank) and each line
    is drawn along its length (position). Returns segment cells + keys and node cells + keys,
    both sorted by key, so 'draw up to progress p' is just a slice."""
    idx = np.arange(nu * nv).reshape(nv, nu)
    iu = sorted(set(range(0, nu, su)) | {nu - 1})
    jv = sorted(set(range(0, nv, sv)) | {nv - 1})
    paths = [idx[:, i] for i in iu] + [idx[j, :] for j in jv]
    key = lambda r, f: 0.9 * r / len(paths) + 0.1 * f
    segs, skeys = [], []
    for r, path in enumerate(paths):
        for k in range(len(path) - 1):
            segs.append((path[k], path[k + 1]))
            skeys.append(key(r, k / (len(path) - 1)))
    nodes = [idx[j, i] for i in iu for j in jv]
    nkeys = np.random.default_rng(len(nodes)).random(len(nodes))      # nodes pop up in random order, before the lines
    so, no = np.argsort(skeys, kind="stable"), np.argsort(nkeys, kind="stable")
    segs, nodes = np.array(segs)[so], np.array(nodes)[no]
    return (np.column_stack([np.full(len(segs), 2), segs]).ravel(), np.array(skeys)[so],
            np.column_stack([np.ones(len(nodes), int), nodes]).ravel(), np.array(nkeys)[no])


def make_scene(pl, elements):
    import pyvista as pv
    items = []
    for el in elements:
        m, pts, rgb = el.material, flat(el.P), (flat(el.C) * 255).astype(np.uint8)
        it = dict(el=el)
        if m.get("points_only"):
            cloud = pv.PolyData(pts.copy())
            cloud["rgb"] = rgb
            it["pts_meshes"] = [cloud]
            it["surf"] = pl.add_mesh(cloud, scalars="rgb", rgb=True, style="points", point_size=3,
                                     render_points_as_spheres=True)
        else:
            grid = pv.StructuredGrid(el.P[..., 0], el.P[..., 1], el.P[..., 2])
            grid["rgb"] = rgb
            it["surf"] = pl.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True, specular=0.3,
                                     specular_power=20, ambient=0.35)
            it["seg_cells"], it["seg_keys"], it["node_cells"], it["node_keys"] = isolines(
                el.u_grid, el.v_grid, *m.get("wire_stride", (6, 4)))
            wire = pv.PolyData(pts.copy(), lines=it["seg_cells"])
            it["wire_mesh"] = wire
            it["pts_meshes"] = [grid, wire]
            it["wire"] = pl.add_mesh(wire, color=m.get("wire_color", WIRE_GREEN), line_width=1.0)
            if m.get("dots"):
                it["dots_meshes"], it["dots"] = [], []                 # young / growing / full-size nodes
                for size in DOT_SIZES:
                    dots = pv.PolyData(pts.copy())
                    it["dots_meshes"].append(dots)
                    it["pts_meshes"].append(dots)
                    it["dots"].append(pl.add_mesh(dots, color="white", point_size=size, render_points_as_spheres=True))
        if m.get("rebuild"):                      # one shared point buffer: moving it moves all meshes of the element
            from vtkmodules.util.numpy_support import vtk_to_numpy
            it["vp"] = pv.vtk_points(pts, deep=True)
            for mesh in it["pts_meshes"]:
                mesh.SetPoints(it["vp"])
            it["vp_np"] = vtk_to_numpy(it["vp"].GetData())
        items.append(it)
    return items


DOT_SIZES = (3, 5, 7)   # point size of newborn, growing and full-size wire nodes


def set_opacity(actor, x):
    actor.GetProperty().SetOpacity(float(x))
    actor.SetVisibility(bool(x > 0.01))


def hide(it):
    set_opacity(it["surf"], 0)
    if "wire" in it:
        set_opacity(it["wire"], 0)
    for actor in it.get("dots", []):
        actor.SetVisibility(False)
    it["b"] = None


def update(items, t):
    for it in items:
        el, m = it["el"], it["el"].material
        s = float(smoothstep(el.surf_t0, el.surf_t1, t))
        p = float(np.clip((t - el.wire_t0) / max(el.wire_t1 - el.wire_t0, 1e-9), 0, 1)) if "wire" in it else 0.0
        pd = float(np.clip((t - m["dots_win"][0]) / (m["dots_win"][1] - m["dots_win"][0]), 0, 1)) if "dots_win" in m else p
        if s <= 0 and p <= 0 and pd <= 0:                                           # not started yet: stay hidden
            if not it.get("hidden"):
                hide(it)
                it["hidden"], it["state"] = True, None
            continue
        it["hidden"] = False
        win = el.material.get("rebuild_win")                                        # (0, end): shape depends on t
        tc = float(np.clip(t, *win)) if win else 0.0
        if (tc, s, p, pd) == it.get("state"):                                       # nothing changed: no work
            continue
        it["state"] = (tc, s, p, pd)
        if win:
            it["vp_np"][:] = flat(el.material["rebuild"](tc))
            it["vp"].GetData().Modified()
            it["vp"].Modified()
        set_opacity(it["surf"], s)
        if "wire" not in it:
            continue
        n = int(np.searchsorted(it["seg_keys"], p, side="right")) if p > 0 else 0   # p=0: nothing drawn yet
        if n and n != it.get("n"):
            it["wire_mesh"].lines = it["seg_cells"][:3 * n]
        it["n"] = n
        keep = el.material.get("wire_keep", 0.25)                                   # how much wire stays over the fill
        set_opacity(it["wire"], (1 - (1 - keep) * s) * (n > 0))
        if "dots" in it:
            # nodes are born over dots_win; each grows through DOT_SIZES as it ages
            b = [int(np.searchsorted(it["node_keys"], pd - lag, side="right")) if pd > 0 else 0 for lag in (0.10, 0.04, 0.0)]
            if b != it.get("b"):
                for mesh, actor, lo, hi in zip(it["dots_meshes"], it["dots"], [0] + b[:2], b):
                    if hi > lo:
                        mesh.verts = it["node_cells"][2 * lo:2 * hi]
                    actor.SetVisibility(hi > lo)
                it["b"] = b
            for actor in it["dots"]:
                actor.GetProperty().SetOpacity(1 - (1 - keep) * s)


STAGES = [(0.00, "STAGE 01 // VASCULAR STEMS & LANCEOLATE FOLIAGE"), (0.26, "STAGE 02 // ANDROECIUM ANTHERS & TRILOBED STIGMA"),
          (0.37, "STAGE 03 // 3D TEPAL ISOLINE BLUEPRINT"), (0.62, "STAGE 04 // SUBSURFACE PIGMENTATION & PAPILLAE"),
          (0.90, "STAGE 05 // COMPLETE BOTANICAL SPECIMEN REVEAL")]


def hud(pl, t):
    label = [name for t0, name in STAGES if t >= t0][-1]
    text = label + "\n" + "=" * int(60 * t) + "-" * (60 - int(60 * t))
    if getattr(pl, "_hud_text", None) == text:                # only redraw the text when it changes
        return
    pl._hud_text = text
    pl.add_text("LILIUM ORIENTALIS", position="upper_left", font_size=9, color="white", name="title")
    pl.add_text(text, position="lower_left", font_size=8, color=(1.0, 0.9, 0.8), name="stage")


def run(t_still=None, shot=None):
    import pyvista as pv
    elements = build_plant()
    pollen = make_pollen(elements)
    pl = pv.Plotter(off_screen=shot is not None, window_size=(900, 1100))
    pl.set_background("black")
    pl.enable_lightkit()
    pl.enable_anti_aliasing("msaa")                            # smooth thin lines
    items = make_scene(pl, elements)
    cloud, halo = pv.PolyData(pollen["base"].copy()), pv.PolyData(pollen["base"].copy())   # halo: big faint copy = glow
    for mesh, size in ((cloud, 5), (halo, 16)):
        mesh["rgba"] = pollen_rgba(pollen, np.zeros(len(pollen["base"])))
        pl.add_mesh(mesh, scalars="rgba", rgba=True, point_size=size, render_points_as_spheres=True)
    pl.camera_position = [(0.0, -6.8, 1.3), (0.0, 0.0, 1.15), (0, 0, 1)]
    start = time.perf_counter()

    def set_pollen(t, secs):
        pts, alpha = pollen_at(pollen, t, secs)
        cloud.points = halo.points = pts
        cloud["rgba"], halo["rgba"] = pollen_rgba(pollen, alpha), pollen_rgba(pollen, 0.12 * alpha)

    def frame(*_):
        secs = time.perf_counter() - start
        t = t_still if t_still is not None else min(secs / 16.0, 1.0)
        update(items, t)
        hud(pl, t)
        set_pollen(t, secs)

    t_first = t_still if t_still is not None else 0.0
    update(items, t_first)                                     # before show(): no flash of the full plant
    hud(pl, t_first)
    set_pollen(t_first, 0.0)
    if not shot:                                           # fixed view: mouse does nothing
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
        pl.show(interactive_update=True, auto_close=False)
        pl.iren.interactor.SetInteractorStyle(vtkInteractorStyleUser())
    if shot:
        pl.screenshot(shot)
        return
    closed = []                                            # the window X fires ExitEvent; pl.close() is never called by it
    pl.iren.interactor.AddObserver("ExitEvent", lambda *_: closed.append(True))
    while True:
        frame()
        pl.iren.process_events()
        if closed:                                         # window already gone: rendering now spams shader errors
            break
        pl.render()
        if t_still is not None:
            time.sleep(0.03)
    pl.close()


# ---------------------------------------------------------------- self-check
def check():
    els = build_plant()
    for e in els:
        assert e.P.dtype == np.float32 and e.C.dtype == np.float32, e.kind
        assert e.P.shape == e.C.shape and np.isfinite(e.P).all() and np.isfinite(e.C).all(), e.kind
        assert 0 <= e.wire_t0 <= e.wire_t1 <= 1 and 0 <= e.surf_t0 <= e.surf_t1 <= 1, e.kind
        if "rebuild" in e.material:                       # animated shape must end exactly at the final shape
            assert np.allclose(e.material["rebuild"](1.0), e.P, atol=1e-4), e.kind
    kinds = [e.kind for e in els]
    assert kinds.count("tepal") == 18 and kinds.count("stamen") == 18 and kinds.count("pistil") == 3
    assert kinds.count("bud") == 2 and kinds.count("stem") == 3
    p = make_pollen(els)
    assert pollen_at(p, 0.0, 0.0)[1].max() == 0 and pollen_at(p, 1.0, 0.0)[1].min() > 0.15   # dark at start, all lit at end
    flower = np.concatenate([e.P.reshape(-1, 3) for e in els if e.kind in ("tepal", "stamen", "pistil")])
    for b in (e for e in els if e.kind == "bud"):                     # buds must not pass through a flower
        c = b.P.mean(axis=1)
        r = np.linalg.norm(b.P - c[:, None], axis=-1).max(axis=1)
        assert (np.linalg.norm(c[:, None] - flower[None], axis=-1).min(axis=1) - r).min() > 0.02, "bud hits a flower"
    print(f"ok: {len(els)} elements")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        check()
    else:
        run(float(a[a.index("--t") + 1]) if "--t" in a else None,
            a[a.index("--shot") + 1] if "--shot" in a else None)
