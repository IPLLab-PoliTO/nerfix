"""bmild/nerf adapter: pinned import, quadrature shadow, analytic field, and checks.

The renderer follows Eq. (27) and supports only the single-pass diagnostic.
Derived in part from bmild/nerf, Copyright (c) 2020 bmild, MIT license.
"""

from contextlib import contextmanager
from functools import partial
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import tensorflow as tf

VERSION = tf.__version__
NAME = "nerf"
SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / NAME
COMMIT = "14c55567a6d0fbd75d3fd12b0411f98160ba3237"
METHODS = ("nerf-infinite", "nerf-zero", "nerfix")


def configure(seed):
    """Enable reproducible CPU TensorFlow operations."""
    tf.config.experimental.enable_op_determinism()
    set_seed(seed)


def set_seed(seed):
    tf.random.set_seed(seed)


def load():
    """Import the pinned upstream module after verifying its git revision."""
    if not (SOURCE_DIR / "run_nerf.py").is_file():
        raise RuntimeError("Run git submodule update --init --recursive first")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != COMMIT:
        raise RuntimeError("Unexpected NeRF revision: " + revision)
    sys.path.insert(0, str(SOURCE_DIR))
    return importlib.import_module("run_nerf")


def render_rays(ray_batch, network_fn, network_query_fn, N_samples,
                retraw=False, lindisp=False, perturb=0., N_importance=0,
                network_fine=None, white_bkgd=False, raw_noise_std=0.,
                verbose=False, *, method="nerfix"):
    """Drop-in single-pass renderer with explicit physical cell widths.

    NeRF baselines retain upstream's endpoint half-cells and adjacent-sample
    distances. nerf-zero omits the terminal contribution (zero final distance),
    never a negative distance to a point at coordinate zero. NeRFix uses K+1
    edges, K samples, and the original cell widths as in reference Eq. (27).
    """
    if method not in METHODS:
        raise ValueError("Unknown method: " + method)
    if N_importance != 0:
        raise NotImplementedError("Figure 3 uses N_importance=0")
    if N_samples < 2:
        raise ValueError("N_samples must be at least 2")
    ray_batch = tf.convert_to_tensor(ray_batch)
    dtype = ray_batch.dtype
    rays_o, rays_d = ray_batch[:, :3], ray_batch[:, 3:6]
    near, far = ray_batch[:, 6:7], ray_batch[:, 7:8]
    tf.debugging.assert_greater(far, near)
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    count = N_samples + 1 if method == "nerfix" else N_samples
    t = tf.linspace(tf.constant(0., dtype), tf.constant(1., dtype), count)
    if lindisp:
        tf.debugging.assert_positive(near)
        nodes = 1. / ((1. - t) / near + t / far)
    else:
        nodes = near * (1. - t) + far * t
    if method == "nerfix":
        lower, upper = nodes[..., :-1], nodes[..., 1:]
        # Widths are computed BEFORE sampling and remain unchanged afterward.
        dists = upper - lower
        u = tf.random.uniform(tf.shape(lower), dtype=dtype) if perturb > 0 else .5
        z_vals = lower + (upper - lower) * u
    else:
        z_vals = nodes
        if perturb > 0:
            mids = .5 * (nodes[..., 1:] + nodes[..., :-1])
            lower = tf.concat([nodes[..., :1], mids], -1)
            upper = tf.concat([mids, nodes[..., -1:]], -1)
            z_vals = lower + (upper - lower) * tf.random.uniform(
                tf.shape(lower), dtype=dtype)
        tail = 1e10 if method == "nerf-infinite" else 0.
        dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1],
                           tf.fill(tf.shape(z_vals[..., :1]), tf.cast(tail, dtype))], -1)
    dists *= tf.linalg.norm(rays_d, axis=-1, keepdims=True)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)
    noise = (tf.random.normal(tf.shape(raw[..., 3]), dtype=dtype) * raw_noise_std
             if raw_noise_std > 0 else 0.)
    alpha = -tf.math.expm1(-tf.nn.relu(raw[..., 3] + noise) * dists)
    # Keep upstream's compositing epsilon; expm1 reduces cancellation.
    weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)
    rgb_map = tf.reduce_sum(weights[..., None] * tf.math.sigmoid(raw[..., :3]), -2)
    acc_map = tf.reduce_sum(weights, -1)
    depth = tf.reduce_sum(weights * z_vals, -1)
    disp_map = 1. / tf.maximum(tf.cast(1e-10, dtype), tf.math.divide_no_nan(depth, acc_map))
    if white_bkgd:
        rgb_map += 1. - acc_map[..., None]
    result = dict(rgb_map=rgb_map, acc_map=acc_map, disp_map=disp_map)
    if retraw:
        result["raw"] = raw
    for key, value in result.items():
        tf.debugging.check_numerics(value, key)
    return result


def analytic_query(pts, viewdirs, network_fn):
    """rho(x)=cos(pi*x/2), kappa=1/2; a unit ray along the x axis."""
    density = tf.cos(tf.constant(np.pi / 2, pts.dtype) * pts[..., :1])
    return tf.concat([tf.zeros_like(pts), density], -1)


def rays(n, dtype=tf.float64):
    return tf.tile(tf.constant([[0., 0., 0., 1., 0., 0., 0., 1.]], dtype), [n, 1])


@contextmanager
def use_method(upstream, method):
    """Yield a batch renderer and restore the upstream function on exit."""
    if method not in METHODS:
        raise ValueError("Unknown NeRF method: " + method)
    original = upstream.render_rays
    try:
        upstream.render_rays = partial(render_rays, method=method)
        yield partial(render_batch, upstream)
    finally:
        upstream.render_rays = original


def render_batch(upstream, num_rays, num_samples):
    """Render the analytic unit ray; return one intensity per trial."""
    return upstream.batchify_rays(
        rays(num_rays), chunk=num_rays, network_fn=None,
        network_query_fn=analytic_query, N_samples=num_samples,
        perturb=1., N_importance=0)["rgb_map"][:, 0].numpy()


def self_check(upstream):
    """Check baseline parity, exact constant density, and midpoint order."""
    original = upstream.render_rays
    batch = rays(8, tf.float32)
    args = dict(network_fn=None, network_query_fn=analytic_query, N_samples=32)
    actual = render_rays(batch, **args, method="nerf-infinite")
    expected = original(batch, **args)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-5, atol=2e-6)

    def constant_query(pts, viewdirs, network_fn):
        return tf.concat([tf.zeros_like(pts), tf.ones_like(pts[..., :1])], -1)

    for perturb in (0., 1.):
        out = render_rays(rays(16), None, constant_query, 32, perturb=perturb)
        np.testing.assert_allclose(out["rgb_map"], .5 * -np.expm1(-1.), atol=2e-9)
    exact = .5 * -np.expm1(-2 / np.pi)
    errors = [abs(float(render_rays(rays(1), None, analytic_query, k)
                        ["rgb_map"][0, 0]) - exact) for k in (8, 16, 32)]
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4., rtol=.02)
    with use_method(upstream, "nerfix"):
        result = upstream.batchify_rays(rays(5), chunk=2, **args)
        assert result["rgb_map"].shape == (5, 3)
    assert upstream.render_rays is original
    print("Checks passed: upstream parity, constant-density exactness, midpoint order, shadow dispatch")
