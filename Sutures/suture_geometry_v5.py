"""
suture_geometry_v5.py
=====================
Focused production engine — 4 validated geometry types only.

Types
-----
  sinusoidal   smooth wave baseline
  jigsaw       rounded bulb-neck-bulb puzzle-piece (restored from v3)
  irregular    freeform biologically inspired (free x and y)
  random       fully free x and y (maximum CNN diversity)

Primary design variables
------------------------
  N     : number of control points  (complexity)
  k     : spline order 2/3/4        (smoothness)
  alpha : amplitude fraction        (tooth depth)

Spline equation used
--------------------
Chord-length parameterised B-spline via scipy.interpolate.make_interp_spline.
The curve C(t) is computed as:
    C(t) = sum_i [ N_i,k(t) * P_i ]
where P_i are the control points and N_i,k(t) are the B-spline basis
functions of order k computed by the Cox-de Boor recursion.
Parameterisation uses chord-length (arc-distance between control points)
rather than uniform t, giving better shape quality for non-uniform spacing.

Physics constraints enforced
-----------------------------
1. No self-intersection
2. Domain bounds [0,S] x [0,S]
3. Minimum radius of curvature >= R_MIN_FRAC * S
4. No zero-length segments
5. Clamped endpoints at (cx,0) and (cx,S) — FEA boundary condition

Author  : Hussin Fekry Abdelrazik  KFUPM Bioengineering 202427940
Advisor : Dr. Ahmed S. Dalaq
"""

import os, csv, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEO_TYPES     = ["sinusoidal", "jigsaw", "irregular", "freeform"]
IMAGE_RES     = 256
CURVE_STEPS   = 800
LINE_WIDTH_PX = 3
R_MIN_FRAC    = 0.010   # min radius of curvature as fraction of S


# ===========================================================================
# PART 1 — SPLINE FITTING  (shared by all types)
# ===========================================================================

def _fit_spline(ctrl_pts: np.ndarray, k: int, S: float) -> np.ndarray:
    """
    Fit a B-spline of order k through ctrl_pts using chord-length
    parameterisation. Returns CURVE_STEPS points clipped to [0, S].

    Spline equation:
        C(t) = sum_i [ N_{i,k}(t) * P_i ],   t in [0, 1]

    Parameters
    ----------
    ctrl_pts : (N, 2) array of control point coordinates
    k        : spline order (2=quadratic, 3=cubic, 4=quartic)
    S        : domain size (for clipping)
    """
    n     = len(ctrl_pts)
    k_eff = max(1, min(k, n - 1))

    # chord-length parameterisation
    diffs = np.diff(ctrl_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = np.where(dists < 1e-10, 1e-10, dists)
    t     = np.concatenate([[0.0], np.cumsum(dists)])
    t     = t / t[-1]

    # ensure strictly increasing (numerical safety)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-8
    t = t / t[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spl = make_interp_spline(t, ctrl_pts, k=k_eff)

    curve = spl(np.linspace(0.0, 1.0, CURVE_STEPS))
    return np.clip(curve, 0.0, S)


# ===========================================================================
# PART 2 — CONTROL POINT PLACEMENT RULES
# ===========================================================================

def _cp_sinusoidal(N, alpha, S, rng):
    """
    Smooth sine-pattern x-deviations, evenly spaced y.
    x_i = cx + maxd * sin(2*pi*n_cycles*eta_i)
    Family identity: smooth alternating pattern.
    """
    cx       = S / 2
    maxd     = alpha * S / 2
    eta      = np.linspace(0, 1, N)
    n_cycles = max(1, (N - 1) // 4)
    x        = cx + maxd * np.sin(2 * np.pi * n_cycles * eta)
    y        = eta * S
    x[0]     = cx;  x[-1] = cx
    return np.column_stack([x, y])


def _cp_jigsaw(N, alpha, S, rng):
    """
    Bulb-neck-bulb puzzle-piece sequence.
    Family identity: rounded arc head wider than neck.

    Each tooth:
      approach neck -> arc around bulb head (8 arc points) -> leave neck

    The 8 arc points around the bulb are what create the rounded
    puzzle-piece shape visible in the CNN images. This is the key
    feature that was lost in v4 and is now restored.

    n_teeth is derived from N: n_teeth = max(2, N // 6)
    """
    cx      = S / 2
    maxd    = alpha * S / 2
    n_teeth = max(2, N // 6)

    neck    = maxd * rng.uniform(0.25, 0.35)   # narrow throat
    head    = maxd * rng.uniform(0.85, 0.98)   # wide bulb (head > neck)
    head_h  = (S / n_teeth) * rng.uniform(0.25, 0.35)  # bulb height

    pts = [[cx, 0.0]]
    tooth_h = S / n_teeth

    for i in range(n_teeth):
        y0   = i * tooth_h
        ymid = y0 + tooth_h * 0.5
        side = 1 if i % 2 == 0 else -1

        # approach — narrow neck entry
        pts.append([cx + side * neck * 0.5,  y0 + tooth_h * 0.12])
        pts.append([cx + side * neck,         y0 + tooth_h * 0.28])

        # rounded bulb head — 8 arc points
        # arc goes from angle pi to 2*pi (semicircle on the far side)
        for a in range(9):
            angle = np.pi + (a / 8) * np.pi
            hx    = cx + side * head + side * head * 0.55 * np.cos(angle)
            hy    = ymid + head_h * np.sin(angle) * 0.6
            pts.append([hx, hy])

        # departure — narrow neck exit
        pts.append([cx + side * neck,         y0 + tooth_h * 0.72])
        pts.append([cx + side * neck * 0.5,   y0 + tooth_h * 0.88])
        pts.append([cx,                        y0 + tooth_h])

    return np.array(pts)


def _cp_irregular(N, alpha, S, rng):
    """
    Biologically inspired freeform.
    Both x and y are free — mimics cranial sutures, cephalopod septa.
    x alternates sign with 35% random flip probability.
    y positions are sorted random — non-uniform vertical spacing.
    """
    cx   = S / 2
    maxd = alpha * S / 2

    ys      = np.sort(rng.uniform(0.05 * S, 0.95 * S, N - 2))
    ys      = np.concatenate([[0.0], ys, [S]])
    xs      = np.zeros(N)
    xs[0]   = cx;  xs[-1] = cx
    sign    = rng.choice([-1, 1])
    for i in range(1, N - 1):
        if rng.random() < 0.35:
            sign *= -1
        xs[i] = cx + sign * rng.uniform(maxd * 0.3, maxd)
    return np.column_stack([xs, ys])


def _cp_freeform(N, alpha, S, rng):
    """
    Freeform spline-based suture geometry from unconstrained
    control-point distributions.
    Fully free x and y — no morphology bias.
    Maximises CNN training diversity while remaining spline-native.
    """
    cx   = S / 2
    maxd = alpha * S / 2

    ys      = np.sort(rng.uniform(0.05 * S, 0.95 * S, N - 2))
    ys      = np.concatenate([[0.0], ys, [S]])
    xs      = rng.uniform(cx - maxd, cx + maxd, N)
    xs[0]   = cx;  xs[-1] = cx
    return np.column_stack([xs, ys])


_CP_RULE = {
    "sinusoidal": _cp_sinusoidal,
    "jigsaw"    : _cp_jigsaw,
    "irregular" : _cp_irregular,
    "freeform"  : _cp_freeform,
}


# ===========================================================================
# PART 3 — PHYSICS VALIDATOR
# ===========================================================================

def _cross2d(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _segs_intersect(p1, p2, p3, p4):
    d1=_cross2d(p3,p4,p1); d2=_cross2d(p3,p4,p2)
    d3=_cross2d(p1,p2,p3); d4=_cross2d(p1,p2,p4)
    return (((d1>0 and d2<0) or (d1<0 and d2>0)) and
            ((d3>0 and d4<0) or (d3<0 and d4>0)))

def _count_crossings(curve, stride=6):
    pts = curve[::stride];  n = len(pts);  c = 0
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if _segs_intersect(pts[i], pts[i+1], pts[j], pts[j+1]):
                c += 1
    return c

def _min_radius_of_curvature(curve):
    """
    Computes minimum radius of curvature using the Frenet formula:
        kappa = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
        R = 1 / kappa
    Uses 5th percentile to represent the near-minimum robustly.
    """
    dx  = np.gradient(curve[:, 0])
    dy  = np.gradient(curve[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num   = np.abs(dx * ddy - dy * ddx)
    den   = (dx**2 + dy**2) ** 1.5
    den   = np.where(den < 1e-12, 1e-12, den)
    kappa = num / den
    R     = 1.0 / np.where(kappa < 1e-10, 1e-10, kappa)
    return float(np.percentile(R, 5))

def validate_geometry(curve, geo_type, S):
    """
    Physics validator. Returns dict:
      valid     : bool
      reasons   : list of failure strings
      crossings : int
      R_min     : float (mm)
    """
    cx      = S / 2
    reasons = []

    # 1. self-intersection
    crossings = _count_crossings(curve)
    if crossings > 0:
        reasons.append(f"self-intersection ({crossings} crossings)")

    # 2. domain bounds
    if not (np.all(curve >= 0) and np.all(curve <= S)):
        reasons.append("curve exits domain bounds")

    # 3. minimum radius of curvature
    R_min = _min_radius_of_curvature(curve)
    if R_min < R_MIN_FRAC * S:
        reasons.append(f"R_min={R_min:.4f} < threshold {R_MIN_FRAC*S:.4f}")

    # 4. zero-length segments
    seg_lens = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    if np.any(seg_lens < 1e-8):
        reasons.append("zero-length segment detected")

    # 5. clamped endpoints
    if abs(curve[0, 0]  - cx) > 0.05 * S or abs(curve[0, 1])       > 0.05 * S:
        reasons.append("start not clamped to (cx, 0)")
    if abs(curve[-1, 0] - cx) > 0.05 * S or abs(curve[-1, 1] - S)  > 0.05 * S:
        reasons.append("end not clamped to (cx, S)")

    return {
        "valid"    : len(reasons) == 0,
        "reasons"  : reasons,
        "crossings": crossings,
        "R_min"    : round(R_min, 5),
    }


# ===========================================================================
# PART 4 — IMAGE RENDERER  (256x256 binary)
# ===========================================================================

def _render_image(curve, S, res=IMAGE_RES, lw=LINE_WIDTH_PX):
    """White curve on black square canvas."""
    img  = np.zeros((res, res), dtype=np.uint8)
    cols = np.clip((curve[:, 0] / S) * (res - 1), 0, res - 1).astype(np.int32)
    rows = np.clip((1 - curve[:, 1] / S) * (res - 1), 0, res - 1).astype(np.int32)
    px   = np.column_stack([cols, rows]).reshape(-1, 1, 2)
    cv2.polylines(img, [px], False, 255, lw)
    return img


# ===========================================================================
# PART 5 — GEOMETRY PLOT  (square, thesis quality)
# ===========================================================================

def _plot(ctrl_pts, curve, result, S, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_xlim(0, S);  ax.set_ylim(0, S);  ax.set_aspect("equal")

    rect = mpatches.FancyBboxPatch(
        (0, 0), S, S, boxstyle="square,pad=0",
        lw=2.5, edgecolor="black", facecolor="white")
    ax.add_patch(rect)
    ax.axvline(x=S / 2, color="#ccc", linestyle="--", lw=1.0, zorder=1)

    color = "#1a5fb4" if result["valid"] else "#cc3300"
    ax.plot(curve[:, 0], curve[:, 1], color=color,
            lw=2.2, zorder=3, solid_capstyle="round")
    ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1],
               color="#cc0000", s=30, zorder=5,
               edgecolors="white", lw=0.6)
    for i, (x, y) in enumerate(ctrl_pts):
        ax.text(x + S*0.025, y, f"a{i+1}", fontsize=6,
                color="#333", va="center", zorder=6)

    status = "valid" if result["valid"] else "INVALID"
    ax.set_title(
        f"{result['geo_type'].capitalize()}  "
        f"N={result['N']}  k={result['k']}  α={result['alpha']:.2f}  "
        f"[{status}]",
        fontsize=8, pad=5)
    ax.set_xlabel("x  (mm)", fontsize=8)
    ax.set_ylabel("y  (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# PART 6 — MAIN CLASS
# ===========================================================================

class SutureGeometry:
    """
    Physics-based freeform spline suture geometry engine — 4 types.

    Parameters
    ----------
    geo_type      : str    one of GEO_TYPES
    N             : int    number of control points (6-20)
    k             : int    spline order {2, 3, 4}
    alpha         : float  amplitude fraction 0.10-0.45
    specimen_size : float  square domain side in mm (default 10.0)
    """

    def __init__(self, geo_type="jigsaw", N=10, k=3,
                 alpha=0.30, specimen_size=10.0):
        assert geo_type in GEO_TYPES, f"geo_type must be one of {GEO_TYPES}"
        assert 4 <= N <= 24,          f"N must be 4-24"
        assert k in (2, 3, 4),        f"k must be 2, 3, or 4"
        assert 0.10 <= alpha <= 0.45, f"alpha must be 0.10-0.45"
        self.geo_type = geo_type
        self.N = N;  self.k = k
        self.alpha = alpha;  self.S = specimen_size

    def generate(self, seed=None):
        """
        Generate one suture geometry.

        Returns dict:
          geo_type, N, k, alpha, valid, reasons, crossings,
          R_min, arc_length, ctrl_pts, curve_array, curve_points, image
        """
        rng      = np.random.default_rng(seed)
        S        = self.S

        ctrl_pts = _CP_RULE[self.geo_type](self.N, self.alpha, S, rng)
        curve    = _fit_spline(ctrl_pts, self.k, S)
        val      = validate_geometry(curve, self.geo_type, S)
        arc      = float(np.sum(np.linalg.norm(
                         np.diff(curve, axis=0), axis=1)))

        return {
            "geo_type"    : self.geo_type,
            "N"           : self.N,
            "k"           : self.k,
            "alpha"       : self.alpha,
            "valid"       : val["valid"],
            "reasons"     : val["reasons"],
            "crossings"   : val["crossings"],
            "R_min"       : val["R_min"],
            "arc_length"  : round(arc, 5),
            "ctrl_pts"    : ctrl_pts,
            "curve_array" : curve,
            "curve_points": [(float(x), float(y)) for x, y in curve],
            "image"       : _render_image(curve, S),
        }

    def visualize(self, result, save_path=None):
        _plot(result["ctrl_pts"], result["curve_array"],
              result, self.S, save_path=save_path)

    def save_image(self, result, path):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True)
        cv2.imwrite(path, result["image"])

    def save_csv_row(self, result, csv_path, image_path="", extra=None):
        extra = extra or {}
        row = {
            "geo_type"  : result["geo_type"],
            "N"         : result["N"],
            "k"         : result["k"],
            "alpha"     : result["alpha"],
            "valid"     : int(result["valid"]),
            "crossings" : result["crossings"],
            "R_min"     : result["R_min"],
            "arc_length": result["arc_length"],
            "image_path": image_path,
            "Fmax"      : None,
            "k_stiff"   : None,
            "U_energy"  : None,
            "sigma_max" : None,
        }
        row.update(extra)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)


# ===========================================================================
# PART 7 — BATCH DATASET GENERATOR
# ===========================================================================

def generate_dataset(output_dir, geo_type, n_samples,
                     N_range=(8, 14), k_values=(2, 3, 4),
                     alpha_range=(0.20, 0.40),
                     specimen_size=10.0, seed=0,
                     skip_invalid=True, verbose=True):
    """
    Generate n_samples geometries of one type.
    N, k, alpha randomised within ranges per geometry.
    Returns path to CSV.
    """
    img_dir  = os.path.join(output_dir, "images")
    csv_path = os.path.join(output_dir, "dataset.csv")
    os.makedirs(img_dir, exist_ok=True)

    rng      = np.random.default_rng(seed)
    saved    = 0
    skipped  = 0
    attempts = 0
    max_att  = n_samples * 12

    while saved < n_samples and attempts < max_att:
        attempts += 1
        N     = int(rng.integers(N_range[0], N_range[1] + 1))
        k     = int(rng.choice(list(k_values)))
        alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))
        s     = int(rng.integers(0, 2**31))

        geo    = SutureGeometry(geo_type=geo_type, N=N, k=k,
                                alpha=alpha, specimen_size=specimen_size)
        result = geo.generate(seed=s)

        if skip_invalid and not result["valid"]:
            skipped += 1
            continue

        img_name = f"{geo_type}_{saved:05d}_N{N}_k{k}.png"
        img_path = os.path.join(img_dir, img_name)
        geo.save_image(result, img_path)
        geo.save_csv_row(result, csv_path, image_path=img_path)
        saved += 1

        if verbose and saved % 50 == 0:
            print(f"    {geo_type}: {saved}/{n_samples}  "
                  f"(skipped invalid: {skipped})")

    if verbose:
        print(f"  ✓ {geo_type:12s}: {saved:4d} saved  |  "
              f"{skipped:3d} invalid  |  {attempts:4d} attempts")
    return csv_path


# ===========================================================================
# SMOKE TEST
# ===========================================================================

if __name__ == "__main__":
    import shutil, pandas as pd
    OUT = "/tmp/suture_v5_test"
    if os.path.exists(OUT): shutil.rmtree(OUT)

    print("=" * 60)
    print("  suture_geometry_v5.py  —  smoke test")
    print("=" * 60)

    test_cfg = {
        "sinusoidal": dict(N=9,  k=3, alpha=0.28),
        "jigsaw"    : dict(N=10, k=3, alpha=0.33),
        "irregular" : dict(N=12, k=4, alpha=0.30),
        "freeform"  : dict(N=10, k=3, alpha=0.28),
    }

    for gtype, cfg in test_cfg.items():
        for seed in range(20):
            geo    = SutureGeometry(geo_type=gtype, **cfg)
            result = geo.generate(seed=seed)
            if result["valid"]:
                geo.save_image(result, f"{OUT}/images/{gtype}.png")
                geo.visualize(result,  save_path=f"{OUT}/plots/{gtype}.png")
                geo.save_csv_row(result, f"{OUT}/dataset.csv",
                                 image_path=f"{OUT}/images/{gtype}.png")
                print(f"  ✓ {gtype:12s}  N={cfg['N']}  k={cfg['k']}  "
                      f"α={cfg['alpha']}  seed={seed}  "
                      f"valid=True  R_min={result['R_min']:.3f}  "
                      f"arc={result['arc_length']:.2f}")
                break
        else:
            print(f"  ✗ {gtype} — no valid geometry in 20 seeds")

    print("\nMini batch (50 geometries per type) ...")
    all_csv = []
    for gtype in GEO_TYPES:
        p = generate_dataset(
            f"{OUT}/batch/{gtype}", gtype, 50,
            seed=7, verbose=False)
        all_csv.append(p)

    master = pd.concat(
        [pd.read_csv(p) for p in all_csv], ignore_index=True)
    print(f"  Total rows : {len(master)}")
    print(f"  Valid      : {master['valid'].sum()}/{len(master)}")
    print(f"  k dist     : {master['k'].value_counts().to_dict()}")
    print(f"  R_min range: {master['R_min'].min():.4f}–"
          f"{master['R_min'].max():.4f} mm")
    print("\nAll tests passed. ✓")
