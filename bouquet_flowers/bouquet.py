"""Blumenstrauss - procedural bouquet that builds itself up from below (NumPy + PyVista).

    python bouquet.py                    animation (about 20 s), wrapped in paper with a ribbon
    python bouquet.py --no-wrap          same bouquet without paper (only the ribbon around the stems)
    python bouquet.py --t 0.6            still frame at time t (0..1)
    python bouquet.py --shot a.png --t 1 save a frame, no window
    python bouquet.py --check            self-check of the geometry, no window
    python bouquet.py --video b.webm     render the animation into a video (smooth in any browser), no window

Stages (global time t from 0 to 1, 1 s = 1/20), camera from the side, rising to 70 deg above during 0.20-0.85:
  0.00-0.20 stems rise from the bottom, centre first; 0.05-0.30 eucalyptus wreath around the rim
  0.20-0.62 heads open kind by kind: lilies, gerberas, chrysanthemums, spray roses (one after another within each kind);
            every 2nd flower releases sparkles that drift away
  0.48-0.65 berries pop up between the spray roses     0.58-0.80 pink paper (higher at the back), ribbon
  0.85-1.00 the camera turns 8 degrees around the finished bouquet
Reuses the lily, the stem/leaf builders and the renderer from ../lilium_orientalis.py.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lilium_orientalis import (GREEN_HIGH, GREEN_LOW, WIRE_GREEN, WIRE_PINK, WIRE_WHITE, element, flower_frame,  # noqa: E402
                               lerp, make_flower, make_leaf, make_scene, make_tube, normalize, pollen_rgba,
                               rising, smoothstep, to_world, update, bezier)

DURATION = 20.0                      # seconds for t = 0 .. 1
DOME_C, DOME_R = np.array([0.0, 0.0, 1.45]), 1.3   # flower heads sit on this sphere cap
Z_TIE, Z_BOTTOM = 0.55, -0.45        # where the stems are bound together / where they are cut
WIRE_ROSE = (1.0, 0.75, 0.82)

ROSE_COLORS = {"pink": ((0.70, 0.10, 0.30), (1.00, 0.64, 0.75)),   # (deep inside, light edge)
               "peach": ((0.90, 0.38, 0.25), (1.00, 0.84, 0.68)),
               "red": ((0.40, 0.00, 0.06), (0.88, 0.10, 0.18)),
               "cream": ((0.90, 0.76, 0.58), (1.00, 0.98, 0.93)),
               "burgundy": ((0.28, 0.00, 0.05), (0.62, 0.04, 0.14))}

# top view of the drawn plan: x right, y = back, distance 1 = 50 deg from the dome top. Front is y < 0.
# (kind, x, y, size): concentric hexagonal rings (roses / chrysanthemums + lilies / gerberas), mirror pairs about
# the same size; sizes from a shrink solver so that no two heads cut into each other, a little jitter left in on purpose
PLAN = [("rose", 0.00, 0.01, 0.42), ("rose", -0.12, 0.22, 0.41), ("rose", 0.12, 0.22, 0.39),
        ("rose", -0.25, 0.00, 0.41), ("rose", 0.26, -0.01, 0.43), ("rose", -0.11, -0.23, 0.40),
        ("rose", 0.13, -0.23, 0.40), ("chrys", -0.27, 0.44, 0.43), ("chrys", 0.28, 0.44, 0.42),
        ("chrys", -0.52, 0.01, 0.43), ("chrys", 0.51, -0.01, 0.42), ("chrys", -0.27, -0.45, 0.44),
        ("chrys", 0.25, -0.45, 0.43), ("lily", 0.00, 0.59, 0.41), ("lily", -0.54, 0.30, 0.40),
        ("lily", 0.54, 0.30, 0.38), ("lily", -0.54, -0.30, 0.39), ("lily", 0.54, -0.30, 0.40),
        ("lily", 0.00, -0.59, 0.42), ("gerbera", -0.38, 0.63, 0.71), ("gerbera", 0.38, 0.64, 0.75),
        ("gerbera", -0.73, -0.03, 0.68), ("gerbera", 0.73, -0.03, 0.67), ("gerbera", -0.39, -0.66, 0.69),
        ("gerbera", 0.38, -0.67, 0.70)]
BERRIES = [(0.00, 0.40), (-0.34, 0.20), (0.34, 0.20), (-0.34, -0.19), (0.34, -0.19), (0.00, -0.40)]
LIFT = {"rose": 0.05, "chrys": 0.15, "lily": -0.12, "gerbera": 0.06}     # radial offset from the dome per kind


def plan_dir(x, y):
    """(theta, phi) on the dome for a point of the plan."""
    return 50 * np.hypot(x, y), np.degrees(np.arctan2(y, x))


def make_layout():
    """Heads from PLAN, centre first (stems rise from the middle outward)."""
    heads = sorted(PLAN, key=lambda h: np.hypot(h[1], h[2]))
    return [(kind, *plan_dir(x, y), size, None) for kind, x, y, size in heads]


LAYOUT = make_layout()


# ---------------------------------------------------------------- helpers
def head_dir(theta, phi, tilt=10.0):
    """Unit vector on the dome, tipped toward the camera (-y) by `tilt` degrees."""
    th, ph, ti = np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(tilt)
    x, y, z = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
    return np.array([x, y * np.cos(ti) - z * np.sin(ti), y * np.sin(ti) + z * np.cos(ti)])


def retime(els, a, b, src=(0.26, 0.84)):
    """Squeeze elements built on the lily's own timeline (src) into the slot a .. b of the bouquet."""
    k = (b - a) / (src[1] - src[0])
    off = a - src[0] * k
    f = lambda x: off + k * x
    for e in els:
        if e.wire_t1 > 0:
            e.wire_t0, e.wire_t1 = f(e.wire_t0), f(e.wire_t1)
        e.surf_t0, e.surf_t1 = f(e.surf_t0), f(e.surf_t1)
        m = e.material
        if "rebuild" in m:
            m["rebuild"] = lambda t, g=m["rebuild"]: g((t - off) / k)
            m["rebuild_win"] = tuple(map(f, m["rebuild_win"]))
        if "dots_win" in m:
            m["dots_win"] = tuple(map(f, m["dots_win"]))
    return els


def pop(P, origin, t0, t1):
    """Shape at time t: grows out of `origin` (scale 0 -> 1) during t0 .. t1."""
    return lambda t: origin + (P - origin) * smoothstep(t0, t1, t)


def stem_line(bind, top, axis, n=48):
    """Centerline from the cut end below the tie, through the tie, bending into the flower axis at the top.
    Resampled to even spacing, so the rising tip moves at a steady speed."""
    up = normalize((top - bind) * [0.22, 0.22, 1.0])
    bottom = bind - up * (bind[2] - Z_BOTTOM) / up[2]                # below the tie the stems cross over
    L = np.linalg.norm(top - bind)
    upper = bezier(np.array([bind, bind + up * 0.30 * L, top - axis * 0.25 * L, top]), np.linspace(0, 1, 60))
    pts = np.vstack([np.linspace(bottom, bind, 16)[:-1], upper])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    q = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(q, s, pts[:, i]) for i in range(3)])


def make_stem_c(center, t0, dur, r0=0.02):
    """Like the lily stem, but along any centerline: wireframe rises from the bottom, then fills in."""
    u = np.linspace(0, 1, len(center))
    P, frames = make_tube(center, r0 * (1 - 0.35 * u), 8)
    C = np.broadcast_to(lerp(GREEN_LOW, GREEN_HIGH, smoothstep(0.3, 1, u))[:, None, :], P.shape)
    return element("stem", P, C, (t0, t0 + 0.03), (t0 + 0.5 * dur, t0 + dur + 0.06), wire_stride=(8, 2),
                   wire_color=WIRE_GREEN, wire_keep=0.4, rebuild=rising(P, t0, dur), rebuild_win=(t0, t0 + dur)), frames


def bind_point(rng):
    return np.array([*rng.normal(0, 0.025, 2), Z_TIE])


# ---------------------------------------------------------------- rose
def rose_local(uu, vv, f, o, s, az):
    """One rose petal in flower coords (z = flower axis). f: 0 innermost .. 1 outermost, o: 0 bud .. 1 open.
    The petal is a piece of a surface of revolution: its profile starts almost upright (a cup) and bends
    away from the axis toward the tip (angle a0 + a1 * u^2), and it wraps around the axis by +-w."""
    mix = lambda bud, full: bud + (full - bud) * o
    a0 = mix(-0.05, -0.14 + 0.42 * f ** 1.5)          # inner petals lean in (closed centre), outer ones open up
    a1 = mix(0.10, 0.20 + 1.25 * f ** 1.8)            # only the outermost tips curl back: a full, cupped bloom
    L = s * mix(0.17 + 0.08 * f, 0.30 + 0.14 * f)     # inner petals taller: the rose has depth
    r0 = s * mix(0.010 + 0.03 * f, 0.015 + 0.06 * f)  # each petal starts a bit further out: they nest
    w = 1.35 - 0.55 * f
    ue = uu * (1 - 0.15 * vv ** 2)                    # rounded upper edge
    alpha = a0 + a1 * ue ** 2
    ds = np.diff(ue, axis=0, prepend=0)
    x = r0 + L * np.cumsum(np.sin(alpha) * ds, axis=0)
    z = L * np.cumsum(np.cos(alpha) * ds, axis=0)
    roll = s * 0.05 * o * f * smoothstep(0.7, 1, ue) ** 2 * (0.4 + 0.6 * np.abs(vv))   # edge rolls outward
    x, z = np.maximum(x + roll, 0.004 * s), z - 0.5 * roll - 0.06 * s * f
    phi = az + vv * w
    return np.stack([x * np.cos(phi), x * np.sin(phi), z], axis=-1)


def make_rose(c, axis, s, colors, t0, dur, n=20, seed=0):
    """Green receptacle, closed bud wireframe, then the petals open from the outside in and fill with color."""
    frame, rng, out = flower_frame(axis), np.random.default_rng(seed), []
    deep, light = np.clip(np.array(colors) * rng.uniform(0.92, 1.05, (2, 3)), 0, 1)   # no two roses exactly alike
    u = np.linspace(0, 1, 10)                                          # receptacle: green bulb under the petals
    P, _ = make_tube(c + np.outer(0.10 * s * u - 0.08 * s, frame[2]), 0.075 * s * np.sin(np.pi * np.clip(u, 0.02, 0.98)), 13)
    out.append(element("receptacle", P, np.broadcast_to(lerp(GREEN_LOW, GREEN_HIGH, u)[:, None, :], P.shape),
                       (t0, t0 + 0.02), (t0 + 0.02, t0 + 0.06), wire_stride=(2, 3), wire_color=WIRE_GREEN, wire_keep=0.0,
                       rebuild=pop(P, c, t0, t0 + 0.04), rebuild_win=(t0, t0 + 0.04)))
    uu, vv = np.meshgrid(np.linspace(0, 1, 14), np.linspace(-1, 1, 11), indexing="ij")
    for i in range(n):
        f = i / (n - 1)
        az = i * np.deg2rad(137.5) + rng.normal(0, 0.1)
        o0 = t0 + 0.30 * dur * (1 - f)
        o1 = o0 + 0.70 * dur
        at = lambda t, f=f, az=az, o0=o0, o1=o1: to_world(rose_local(uu, vv, f, smoothstep(o0, o1, t), s, az), c, frame)
        shade = smoothstep(-0.2, 1.1, 0.8 * uu + 0.3 * f + 0.2 * np.abs(vv))
        col = lerp(deep, light, shade) * (0.8 + 0.2 * f)              # the heart of the rose is darker
        out.append(element("petal", at(1.0), col, (t0, t0 + 0.25 * dur), (o0 + 0.4 * (o1 - o0), o1),
                           wire_stride=(3, 2), wire_color=WIRE_ROSE, wire_keep=0.0, rebuild=at, rebuild_win=(t0, o1)))
    return out


# ---------------------------------------------------------------- daisy
def ray_local(uu, vv, L, W, a0, a1, r0):
    """Daisy ray: narrow strap with a rounded tip, bending away from the axis, slight trough down the middle."""
    alpha = a0 + a1 * uu
    x = r0 + L * (np.cos(a0) - np.cos(alpha)) / a1
    z = L * (np.sin(alpha) - np.sin(a0)) / a1
    y = vv * W * np.sqrt(np.clip(1 - uu ** 6, 0, 1)) * (0.55 + 0.45 * np.sqrt(uu))
    return np.stack([x, y, z + 0.25 * y ** 2 / W], axis=-1)


def make_daisy(c, axis, s, t0, dur, rows=((8, 1.0, 1.42, 0.40), (8, 0.92, 1.52, 0.40)), W=0.055, disc_r=0.095,
               ray_col=((0.85, 0.90, 0.55), (1.0, 1.0, 0.98)), disc_col=((1.0, 0.80, 0.15), (0.85, 0.45, 0.05)), seed=0):
    """Disc pops up first, then the rays unfold one after another around the circle, outer row first.
    rows: (count, length, opening angle from the axis, tip curl) per row."""
    frame, rng, out = flower_frame(axis), np.random.default_rng(seed), []
    Z = frame[2]
    u = np.linspace(0, 1, 10)
    P, _ = make_tube(c + np.outer(0.055 * s * u - 0.01 * s, Z), disc_r * s * np.sqrt(np.clip(1 - u ** 2, 0, 1)) + 1e-4, 17)
    col = lerp(*disc_col, smoothstep(0.35, 1.0, u))
    d1 = t0 + 0.25 * dur
    out.append(element("disc", P, np.broadcast_to(col[:, None, :], P.shape), (t0, t0 + 0.05), (t0 + 0.1 * dur, d1),
                       wire_stride=(2, 2), wire_color=(1.0, 0.9, 0.4), wire_keep=0.0, rebuild=pop(P, c, t0, d1),
                       rebuild_win=(t0, d1)))
    uu, vv = np.meshgrid(np.linspace(0, 1, 12), np.linspace(-1, 1, 7), indexing="ij")
    col = lerp(*ray_col, smoothstep(0.0, 0.3, uu))
    for r, (n, ls, a_open, curl) in enumerate(rows):
        for k in range(n):
            az = 2 * np.pi * (k + 0.5 * r) / n + rng.normal(0, 0.05)
            o0 = t0 + dur * (0.10 + 0.30 * k / n + 0.12 * r)
            o1 = o0 + 0.40 * dur

            def at(t, az=az, r=r, ls=ls, a_open=a_open, curl=curl, o0=o0, o1=o1):
                o = smoothstep(o0, o1, t)
                mix = lambda bud, full: bud + (full - bud) * o
                loc = ray_local(uu, vv, s * mix(0.16, 0.36 * ls), W * s, mix(0.25, a_open), mix(0.08, curl), 0.75 * disc_r * s)
                ca, sa = np.cos(az), np.sin(az)
                loc = np.stack([ca * loc[..., 0] - sa * loc[..., 1], sa * loc[..., 0] + ca * loc[..., 1],
                                loc[..., 2] - 0.008 * s * r], axis=-1)
                return to_world(loc, c, frame)
            out.append(element("ray", at(1.0), col, (o0 - 0.02, o0 + 0.03), (o0 + 0.5 * (o1 - o0), o1 + 0.02),
                               wire_stride=(3, 3), wire_color=WIRE_WHITE, wire_keep=0.0, rebuild=at, rebuild_win=(o0 - 0.02, o1)))
    return out


def make_gerbera(c, axis, s, t0, dur, seed=0):
    """Coral-pink gerbera: three rows of narrow rays around a dark green-brown eye."""
    return make_daisy(c, axis, s, t0, dur, rows=((20, 1.0, 1.45, 0.30), (20, 0.85, 1.30, 0.30), (14, 0.55, 1.00, 0.20)),
                      W=0.035, disc_r=0.11, ray_col=((0.80, 0.25, 0.32), (1.0, 0.52, 0.54)),
                      disc_col=((0.50, 0.48, 0.20), (0.28, 0.16, 0.05)), seed=seed)


def make_chrys(c, axis, s, t0, dur, seed=0):
    """White chrysanthemum: two rows of narrow white rays around a yellow centre."""
    return make_daisy(c, axis, s, t0, dur, rows=((20, 1.0, 1.40, 0.30), (18, 0.85, 1.15, 0.20)),
                      W=0.04, disc_r=0.085, ray_col=((0.85, 0.90, 0.70), (1.0, 1.0, 0.97)),
                      disc_col=((0.95, 0.85, 0.35), (0.85, 0.62, 0.15)), seed=seed)


# ---------------------------------------------------------------- greens and fillers
def make_eucalyptus(center, frames, t0, dur, n=9, seed=0):
    """Round blue-green leaves in pairs along a branch; each pops out when the rising tip passes it."""
    T, N, B = frames
    rng, out = np.random.default_rng(seed), []
    uu, vv = np.meshgrid(np.linspace(0, 1, 9), np.linspace(-1, 1, 7), indexing="ij")
    nu = len(center)
    for j in range(n):
        i = int((0.72 + 0.27 * j / (n - 1)) * (nu - 1))                 # outer part only: a wreath, not a clump
        a = j * np.pi + 0.6 * j + rng.normal(0, 0.2)
        d = normalize(np.cos(a) * N[i] + np.sin(a) * B[i])
        ld = normalize(d + 0.45 * T[i])                                # leaves point outward and a bit up
        e = normalize(np.cross(ld, T[i]))
        nrm = np.cross(ld, e)
        R = 0.13 * (1 - 0.45 * j / n)                                  # smaller toward the tip
        along, hw = R * (1 - np.cos(np.pi * uu)), 0.85 * R * np.sin(np.pi * uu)
        P = center[i] + along[..., None] * ld + (hw * vv)[..., None] * e + (0.2 * R * vv ** 2)[..., None] * nrm
        col = lerp((0.28, 0.44, 0.42), (0.60, 0.74, 0.68), np.clip(0.3 + 0.7 * np.abs(vv) * uu, 0, 1))
        tl = t0 + dur * i / (nu - 1)
        out.append(element("euca", P, col, (tl, tl + 0.03), (tl + 0.02, tl + 0.06), wire_stride=(2, 2),
                           wire_color=(0.75, 0.95, 0.85), wire_keep=0.0, rebuild=pop(P, center[i], tl, tl + 0.04),
                           rebuild_win=(tl, tl + 0.04)))
    return out


def make_berries(c, axis, t0, dur, seed=0):
    """Hypericum: a little cluster of glossy dark red berries, growing out of the cluster centre."""
    rng = np.random.default_rng(seed)
    X, Y, Z = flower_frame(axis)
    pts = (c + rng.normal(0, 0.09, (18, 1)) * X + rng.normal(0, 0.09, (18, 1)) * Y + rng.uniform(0, 0.08, (18, 1)) * Z)[:, None, :]
    col = lerp((0.35, 0.02, 0.06), (0.70, 0.10, 0.14), rng.random(18))[:, None, :]
    return element("berry", pts, col, (0, 0), (t0, t0 + dur), points_only=True,
                   rebuild=pop(pts, c, t0, t0 + dur), rebuild_win=(t0, t0 + dur))


# ---------------------------------------------------------------- paper and ribbon
def paper_radius(z, r_tie, k=0.85):
    h = z - Z_TIE
    return np.where(h > 0, r_tie + k * np.clip(h, 0, None) ** 0.85, r_tie + 0.35 * -h)   # flares a little below the tie


def paper_r(z, az, r_tie, inner=False):
    """Radius of the paper at height z and azimuth az. The back is higher, but the cone flares as if the rim were
    level (zf), so the rim is an even circle seen from above. Folds: 7 around, the inner layer half a fold offset."""
    ruffle = 0.06 * np.sin(13 * az + 1.3) if inner else 0.03 * np.sin(9 * az)
    level = (2.45 if inner else 2.35) + ruffle
    top = level + 0.32 * np.sin(az)
    zf = 0.18 + (z - 0.18) * (level - 0.18) / (top - 0.18)
    folds = 1 + 0.045 * np.sin(7 * az + 1.5 * zf + (np.pi if inner else 0.0)) * smoothstep(Z_TIE, Z_TIE + 0.6, zf)
    return paper_radius(zf, r_tie) * folds * (0.88 if inner else 1.0), top   # 0.88: layers never cross


def make_paper(r_tie, t0, dur, inner=False, nu=40, nv=97):
    """Cone of paper, gathered at the tie, high at the back, low at the front, with soft folds.
    It grows upward (rows above the rising edge collapse onto it), the wireframe first."""
    az = np.linspace(-np.pi / 2, 1.5 * np.pi, nv)                       # starts at the front; last == first
    top = paper_r(0.0, az, r_tie, inner)[1]
    u = np.linspace(0, 1, nu)[:, None]
    z = 0.18 + u * (top[None, :] - 0.18)
    r = paper_r(z, az, r_tie, inner)[0]
    P = np.stack([r * np.cos(az), r * np.sin(az), z], axis=-1)
    crinkle = 0.5 + 0.5 * np.sin(23 * az + 3 * z) * np.sin(5 * az - 2 * z)
    if inner:
        col = lerp((0.92, 0.62, 0.68), (1.0, 0.84, 0.86), np.clip(0.4 * crinkle + 0.6 * u, 0, 1))
    else:
        col = lerp((0.80, 0.44, 0.50), (0.95, 0.64, 0.69), np.clip(0.3 * crinkle + 0.5 * u, 0, 1))
    return element("paper", P, col, (t0, t0 + 0.04), (t0 + 0.5 * dur, t0 + dur + 0.03), wire_stride=(4, 4),
                   wire_color=WIRE_PINK, wire_keep=0.0, rebuild=rising(P, t0, dur),
                   rebuild_win=(t0, t0 + dur))


RIBBON, RIBBON_LIGHT = (0.70, 0.22, 0.36), (0.98, 0.66, 0.76)


def ribbon_el(P, t0, dur, rows=None):
    v = np.linspace(-1, 1, P.shape[1])
    col = np.broadcast_to(lerp(RIBBON, RIBBON_LIGHT, 0.6 * (1 - np.abs(v)) ** 2)[None], P.shape)
    return element("ribbon", P, col, (t0, t0 + 0.02), (t0 + 0.5 * dur, t0 + dur + 0.02), wire_stride=(3, 3),
                   wire_color=WIRE_ROSE, wire_keep=0.0, rebuild=rows or rising(P, t0, dur), rebuild_win=(t0, t0 + dur))


def make_ribbon(r, t0):
    """Satin band around the tie, a bow (figure-eight loops) and two tails with a V-cut at the front."""
    v = np.linspace(-1, 1, 5)
    out = []
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, 49)                        # band: starts and ends at the bow
    band = np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, Z_TIE)], axis=-1)
    out.append(ribbon_el(band[:, None, :] + (0.045 * v)[None, :, None] * [0, 0, 1], t0, 0.03))
    F = np.array([0.0, -r - 0.025, Z_TIE])                              # front centre: where the bow sits
    s = np.linspace(0, 2 * np.pi, 81)
    A, Bh = 0.24, 0.24
    loop = F + np.stack([A * np.sin(s), -0.06 * np.sin(s) ** 2, Bh * np.sin(s) * np.cos(s) + 0.35 * A * np.sin(s) ** 2], axis=-1)
    T = normalize(np.gradient(loop, axis=0))
    wdir = normalize(0.6 * normalize(np.cross(T, [0, 1.0, 0])) + 0.8 * np.array([0, 1.0, 0]))
    out.append(ribbon_el(loop[:, None, :] + (0.05 * v)[None, :, None] * wdir[:, None, :], t0 + 0.03, 0.04))
    uu, vv = np.meshgrid(np.linspace(0, 1, 30), v, indexing="ij")
    q = uu * (1 - 0.12 * (1 - np.abs(vv)))                              # V-notch at the ends
    for side in (-1, 1):
        x = side * (0.03 + 0.15 * q + 0.03 * np.sin(2 * np.pi * q))
        c = F + np.stack([x, -0.01 - 0.03 * q, -0.5 * q], axis=-1)
        n = normalize(np.stack([0.5 + 0.0 * q, 0.25 * np.sin(3 * q), side * 0.15 + 0.0 * q], axis=-1))
        out.append(ribbon_el(c + (0.045 * vv)[..., None] * n, t0 + 0.05, 0.04))
    u = np.linspace(0, 1, 12)
    knot, _ = make_tube(F + np.outer(0.09 * u - 0.045, [1.0, 0, 0]), 0.035 * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) + 0.006, 13)
    out.append(ribbon_el(knot, t0 + 0.07, 0.03, rows=pop(knot, F, t0 + 0.07, t0 + 0.10)))
    return out


# ---------------------------------------------------------------- bouquet
BLOOM = {"lily": (0.20, 0.45), "gerbera": (0.28, 0.50), "chrys": (0.35, 0.58), "rose": (0.42, 0.62)}
HEAD_DUR = {"lily": 0.17, "gerbera": 0.15, "chrys": 0.16, "rose": 0.12}   # how long one head takes to open


def head_times(i):
    """(start, end) of opening for head i: heads of one kind open one after another inside their kind's window."""
    kind = LAYOUT[i][0]
    same = [j for j, h in enumerate(LAYOUT) if h[0] == kind]
    a, b = BLOOM[kind]
    h0 = a + (b - a - HEAD_DUR[kind]) * same.index(i) / max(len(same) - 1, 1)
    return h0, h0 + HEAD_DUR[kind]


def build_bouquet(wrap=True):
    rng = np.random.default_rng(7)
    els = []
    for i, (kind, theta, phi, size, variant) in enumerate(LAYOUT):
        axis = head_dir(theta, phi)
        c = DOME_C + (DOME_R + LIFT[kind]) * axis                          # kinds sit at different heights, like a real bouquet
        h0, h1 = head_times(i)
        st = min(0.10 * i / len(LAYOUT), h0 - 0.10)                    # stems: 0 .. 0.20, centre first
        center = stem_line(bind_point(rng), c - 0.02 * axis, axis)
        stem, frames = make_stem_c(center, st, 0.10)
        els.append(stem)
        if i % 3 == 0:                                                   # a leaf on every third stem, above the tie
            j = int(0.45 * (len(center) - 1))
            els.append(make_leaf(center[j], frames[0][j], frames[1][j], 0.45, 0.09, t0=st + 0.10 * j / (len(center) - 1),
                                 pitch0=0.6, droop=1.0))
        dur = h1 - h0
        n0 = len(els)
        if kind == "lily":
            els += retime(make_flower(c, axis + [0, -0.25, 0], size, seed=i), h0, h0 + dur)
        elif kind == "gerbera":
            els += make_gerbera(c, axis, size, h0, dur, seed=i)
        elif kind == "chrys":
            els += make_chrys(c, axis, size, h0, dur, seed=i)
        else:                                                            # spray rose: one small dark red rose
            els += make_rose(c, axis, size, ROSE_COLORS["burgundy"], h0, dur, n=22, seed=i)
        for e in els[n0:]:
            e.material["head"] = i                                       # which flower a part belongs to (overlap check)
    for j in range(12):                                                 # eucalyptus wreath around the rim: 0.05 .. 0.30
        d = head_dir(46, 270 + 30 * j)
        st = 0.05 + 0.13 * j / 11
        center = stem_line(bind_point(rng), DOME_C + 1.35 * DOME_R * d, d)
        stem, frames = make_stem_c(center, st, 0.12, r0=0.012)
        els.append(stem)
        els += make_eucalyptus(center, frames, st, 0.12, n=10, seed=j)
    for j, (x, y) in enumerate(BERRIES):                                # berries between the spray roses: 0.58 .. 0.75
        d = head_dir(*plan_dir(x, y))
        c = DOME_C + (DOME_R + 0.22) * d
        st = 0.20 + 0.02 * j
        center = stem_line(bind_point(rng), c, d)
        els.append(make_stem_c(center, st, 0.08, r0=0.008)[0])
        els.append(make_berries(c, d, 0.48 + 0.10 * j / (len(BERRIES) - 1), 0.07, seed=j))
    for j, (x, y) in enumerate(BERRIES):                                # eucalyptus bed under the roses: foliage in the gaps
        d = head_dir(*plan_dir(0.7 * x, 0.7 * y))
        st = 0.08 + 0.02 * j
        center = stem_line(bind_point(rng), DOME_C + (DOME_R - 0.12) * d, d)
        stem, frames = make_stem_c(center, st, 0.10, r0=0.010)
        els.append(stem)
        els += make_eucalyptus(center, frames, st, 0.10, n=6, seed=20 + j)
    r_tie = 0.12 if wrap else 0.075
    if wrap:
        els += [make_paper(r_tie, 0.58, 0.10, inner=True), make_paper(r_tie, 0.62, 0.11)]
    els += make_ribbon(r_tie + (0.01 if wrap else 0.0), 0.70)
    return els


def make_sparkles(per_burst=14, seed=3):
    """Every second flower releases a puff of glowing specks the moment it is fully open; they drift away and fade."""
    rng = np.random.default_rng(seed)
    base, vel, birth = [], [], []
    for i, (kind, theta, phi, _, _) in enumerate(LAYOUT[::2]):
        axis = head_dir(theta, phi)
        base.append(DOME_C + DOME_R * axis + rng.normal(0, 0.08, (per_burst, 3)))
        vel.append(normalize(axis + rng.normal(0, 0.6, (per_burst, 3)) + [0, 0, 0.4]) * rng.uniform(2.5, 5.0, (per_burst, 1)))
        birth.append(head_times(2 * i)[1] + rng.uniform(0, 0.02, per_burst))
    n = per_burst * len(base)
    palette = np.array([(1.0, 1.0, 1.0), (1.0, 0.92, 0.6), (1.0, 0.8, 0.9)])
    return dict(base=np.vstack(base), vel=np.vstack(vel), birth=np.concatenate(birth), freq=rng.uniform(0.2, 0.7, (n, 3)),
                ph=rng.uniform(0, 2 * np.pi, (n, 3)), twinkle=rng.uniform(0, 2 * np.pi, n),
                rgb=palette[rng.choice(3, n, p=[0.5, 0.3, 0.2])])


def sparkles_at(p, t, secs):
    """Positions (n, 3) and brightness 0..1: fly out along vel after birth, fade out over ~8 s, twinkle meanwhile."""
    age = np.clip(t - p["birth"], 0, None)
    pts = p["base"] + p["vel"] * age[:, None] + 0.05 * np.sin(p["freq"] * secs + p["ph"])
    alpha = (t >= p["birth"]) * smoothstep(0, 0.01, age) * (1 - smoothstep(0.10, 0.28, age))
    return pts.astype(np.float32), alpha * (0.6 + 0.4 * np.sin(1.7 * secs + p["twinkle"]))


# ---------------------------------------------------------------- render
STAGES = [(0.00, "PHASE 01 // STIELE & EUKALYPTUS WACHSEN"), (0.20, "PHASE 02 // BLUETEN OEFFNEN SICH EINZELN"),
          (0.48, "PHASE 03 // BEEREN"), (0.58, "PHASE 04 // PAPIER & SCHLEIFE"),
          (0.96, "PHASE 05 // FERTIGER BLUMENSTRAUSS")]


def hud(pl, t, wrap):
    stages = STAGES if wrap else [(a, n.replace("PAPIER & ", "")) for a, n in STAGES]
    label = [name for t0, name in stages if t >= t0][-1]
    text = label + "\n" + "=" * int(60 * t) + "-" * (60 - int(60 * t))
    if getattr(pl, "_hud_text", None) == text:
        return
    pl._hud_text = text
    pl.add_text("BLUMENSTRAUSS", position="upper_left", font_size=9, color="white", name="title")
    pl.add_text(text, position="lower_left", font_size=8, color=(1.0, 0.9, 0.8), name="stage")


def camera_at(t):
    """Side view at stem height -> 70 deg from above (t 0.20 .. 0.85), then turn 8 deg around (t 0.85 .. 1)."""
    k = smoothstep(0.20, 0.85, t)
    el = np.deg2rad(6 + 64 * k)
    az = np.deg2rad(-90 + 8 * smoothstep(0.85, 1.0, t))
    focal = np.array([0.0, 0.0, 1.15 + 0.95 * k])          # end: aim at the flowers, not the stems
    pos = focal + 8.8 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return [tuple(pos), tuple(focal), (0, 0, 1)]


def run(t_still=None, shot=None, wrap=True, video=None):
    import pyvista as pv
    elements = build_bouquet(wrap)
    sparkles = make_sparkles()
    pl = pv.Plotter(off_screen=bool(shot or video), window_size=(1000, 1200))
    pl.set_background("black", top=(0.10, 0.05, 0.10))
    pl.enable_lightkit()
    pl.enable_anti_aliasing("msaa")
    pl.enable_ssao(radius=0.15, bias=0.01, kernel_size=64)               # soft contact shadows between the flowers
    items = make_scene(pl, elements)
    for it in items:
        if it["el"].kind == "berry":                                    # berries: bigger glossy dots
            it["surf"].GetProperty().SetPointSize(14)
            it["surf"].GetProperty().SetSpecular(0.6)
    cloud, halo = pv.PolyData(sparkles["base"].copy()), pv.PolyData(sparkles["base"].copy())
    for mesh, size in ((cloud, 5), (halo, 16)):
        mesh["rgba"] = pollen_rgba(sparkles, np.zeros(len(sparkles["base"])))
        pl.add_mesh(mesh, scalars="rgba", rgba=True, point_size=size, render_points_as_spheres=True)

    def set_sparkles(t, secs):
        pts, alpha = sparkles_at(sparkles, t, secs)
        cloud.points = halo.points = pts
        cloud["rgba"], halo["rgba"] = pollen_rgba(sparkles, alpha), pollen_rgba(sparkles, 0.12 * alpha)

    clock = [0.0, 0.0]                                                 # animation seconds, wall clock of the last frame

    def frame():
        now = time.perf_counter()
        clock[0] += min(now - clock[1], 1 / 30)                         # a slow frame slows down, it does not jump
        clock[1] = now
        secs = clock[0]
        t = t_still if t_still is not None else min(secs / DURATION, 1.0)
        update(items, t)
        pl.camera_position = camera_at(t)
        hud(pl, t, wrap)
        set_sparkles(t, secs)

    t_first = t_still if t_still is not None else 0.0
    update(items, t_first)
    pl.camera_position = camera_at(t_first)
    hud(pl, t_first, wrap)
    set_sparkles(t_first, 0.0)
    if shot:
        pl.screenshot(shot)
        return
    if video:                                                          # fixed time step per frame: no stutter possible
        import cv2
        fps, n = 30, int(DURATION * 30)
        out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"VP80"), fps, tuple(pl.window_size))
        for i in range(n + 2 * fps):                                   # + 2 s on the finished bouquet
            t = min(i / n, 1.0)
            update(items, t)
            pl.camera_position = camera_at(t)
            hud(pl, t, wrap)
            set_sparkles(t, i / fps)
            pl.render()
            img = pl.screenshot(return_img=True)
            assert img.shape[1::-1] == tuple(pl.window_size), img.shape  # VideoWriter silently drops wrong-sized frames
            out.write(np.ascontiguousarray(img[..., 2::-1]))           # RGB -> BGR
            print(f"\rframe {i + 1}/{n + 2 * fps}", end="")
        out.release()
        print(f"\nsaved {video}")
        return
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
    pl.show(interactive_update=True, auto_close=False)
    clock[1] = time.perf_counter()                                     # after the window is up: no jump at the start
    pl.iren.interactor.SetInteractorStyle(vtkInteractorStyleUser())   # fixed view: mouse does nothing
    closed = []
    pl.iren.interactor.AddObserver("ExitEvent", lambda *_: closed.append(True))
    while True:
        frame()
        pl.iren.process_events()
        if closed:
            break
        pl.render()
        if t_still is not None:
            time.sleep(0.03)
    pl.close()


# ---------------------------------------------------------------- self-check
def overlapping_heads(els, eps=0.015):
    """Pairs of flower heads whose surfaces come closer than eps (they cut into each other)."""
    heads = {}
    for e in els:
        if "head" in e.material and e.kind != "papillae":
            heads.setdefault(e.material["head"], []).append(e.P.reshape(-1, 3))
    pts = {i: np.vstack(v) for i, v in heads.items()}
    bad = []
    for i in pts:
        for j in pts:
            if j <= i:
                continue
            lo = np.maximum(pts[i].min(0), pts[j].min(0)) - eps                # only the region where both boxes meet
            hi = np.minimum(pts[i].max(0), pts[j].max(0)) + eps
            if (lo > hi).any():
                continue
            a, c = (q[((q >= lo) & (q <= hi)).all(1)] for q in (pts[i], pts[j]))
            if any((((a[k:k + 2000, None] - c[None]) ** 2).sum(-1) < eps ** 2).any() for k in range(0, len(a), 2000)):
                bad.append((i, j))
    return bad


def check():
    for wrap in (True, False):
        els = build_bouquet(wrap)
        for e in els:
            assert e.P.dtype == np.float32 and e.P.shape == e.C.shape, e.kind
            assert np.isfinite(e.P).all() and np.isfinite(e.C).all(), e.kind
            assert 0 <= e.wire_t0 <= e.wire_t1 <= 1 and 0 <= e.surf_t0 <= e.surf_t1 <= 1, (e.kind, e.wire_t0, e.surf_t1)
            if "rebuild" in e.material:                                 # animated shape must end at the final shape
                assert np.allclose(e.material["rebuild"](1.0), e.P, atol=1e-4), e.kind
        bad = overlapping_heads(els)
        assert not bad, f"flowers cut into each other: {[(LAYOUT[i][0], LAYOUT[j][0]) for i, j in bad]}"
        kinds = [e.kind for e in els]
        assert kinds.count("paper") == (2 if wrap else 0) and kinds.count("ribbon") == 5
        heads = np.array([DOME_C + DOME_R * head_dir(th, ph) for _, th, ph, _, _ in LAYOUT])
        gap = np.linalg.norm(heads[:, None] - heads[None], axis=-1) + np.eye(len(heads)) * 9
        if wrap:                                                        # no flower may stick out through the paper
            paper = [e for e in els if e.kind == "paper"][1].P
            rim = paper[-1][np.argsort(np.arctan2(paper[-1, :, 1], paper[-1, :, 0]))]
            for e in els:
                if e.kind in ("paper", "ribbon", "stem", "leaf"):
                    continue
                P = e.P.reshape(-1, 3)
                top = np.interp(np.arctan2(P[:, 1], P[:, 0]), np.arctan2(rim[:, 1], rim[:, 0]), rim[:, 2], period=2 * np.pi)
                wall = paper_r(P[:, 2], np.arctan2(P[:, 1], P[:, 0]), 0.12)[0]
                out = (P[:, 2] < top - 0.02) & (np.hypot(P[:, 0], P[:, 1]) > 0.94 * wall)
                assert not out.any(), f"{e.kind} sticks out through the paper"
        print(f"ok ({'wrapped' if wrap else 'no wrap'}): {len(els)} elements, closest heads {gap.min():.2f} apart")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        check()
    else:
        run(float(a[a.index("--t") + 1]) if "--t" in a else None,
            a[a.index("--shot") + 1] if "--shot" in a else None, wrap="--no-wrap" not in a,
            video=a[a.index("--video") + 1] if "--video" in a else None)
