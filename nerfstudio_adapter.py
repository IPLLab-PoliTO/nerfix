"""NeRFStudio adapter: pinned import, endpoint correction, and analytic rendering.

The correction wraps SpacedSampler.generate_ray_samples without editing its
source. K counts jittered boundaries, giving K-1 midpoint field evaluations.
"""

from contextlib import contextmanager
from functools import partial
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

VERSION = torch.__version__
NAME = "nerfstudio"
SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / NAME
COMMIT = "50e0e3c70c775e89333256213363badbf074f29d"
METHODS = ("nerfstudio", "nerfstudio-fix")


def configure(seed):
    """Keep the upstream grid, jitter, and weights deterministic in float64."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_seed(seed)


def set_seed(seed):
    torch.manual_seed(seed)


def load():
    """Import the pinned upstream module after verifying its git revision."""
    if not (SOURCE_DIR / "nerfstudio/model_components/ray_samplers.py").is_file():
        raise RuntimeError("Run git submodule update --init --recursive first")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError("Unexpected NeRFStudio revision: " + revision)
    sys.path.insert(0, str(SOURCE_DIR))
    return importlib.import_module("nerfstudio.model_components.ray_samplers")


def generate_ray_samples(self, ray_bundle=None, num_samples=None, *, original):
    """Shadow SpacedSampler: retain interior bins, restore both endpoints.

    Calls the unmodified upstream sampler before changing only the first and
    last spacing boundaries. Rebuilds RaySamples so frustums, deltas, spacing
    coordinates and the mapping to physical distances stay consistent.
    """
    samples = original(self, ray_bundle=ray_bundle, num_samples=num_samples)
    starts = samples.spacing_starts.clone()
    ends = samples.spacing_ends.clone()
    starts[..., 0, :] = 0.
    ends[..., -1, :] = 1.
    to_euclidean = samples.spacing_to_euclidean_fn
    return ray_bundle.get_ray_samples(
        bin_starts=to_euclidean(starts[..., 0])[..., None],
        bin_ends=to_euclidean(ends[..., 0])[..., None],
        spacing_starts=starts, spacing_ends=ends,
        spacing_to_euclidean_fn=to_euclidean)


def rays(n, upstream):
    return upstream.RayBundle(
        origins=torch.zeros((n, 3)),
        directions=torch.tensor([[1., 0., 0.]]).repeat(n, 1),
        pixel_area=torch.ones((n, 1)),
        nears=torch.zeros((n, 1)), fars=torch.ones((n, 1)))


def render_batch(upstream, num_rays, num_samples):
    # Eqs. (28)-(30) count K boundary samples and K-1 midpoint evaluations.
    sampler = upstream.UniformSampler(num_samples=num_samples - 1, single_jitter=False)
    sampler.train()
    with torch.no_grad():
        samples = sampler(rays(num_rays, upstream))
        x = samples.frustums.get_positions()[..., :1]
        density = torch.cos((np.pi / 2) * x)
        return (.5 * samples.get_weights(density).sum(dim=-2)[..., 0]).numpy()


@contextmanager
def use_method(upstream, method):
    """Yield a batch renderer with either the original or corrected sampler."""
    if method not in METHODS:
        raise ValueError("Unknown NeRFStudio method: " + method)
    original = upstream.SpacedSampler.generate_ray_samples
    try:
        if method == "nerfstudio-fix":
            # A regular function retains descriptor binding when put on a class.
            def corrected(self, ray_bundle=None, num_samples=None):
                return generate_ray_samples(self, ray_bundle, num_samples, original=original)
            upstream.SpacedSampler.generate_ray_samples = corrected
        yield partial(render_batch, upstream)
    finally:
        upstream.SpacedSampler.generate_ray_samples = original


def self_check(upstream):
    """Check domain coverage, unchanged interiors, and quadrature accuracy."""
    exact = .5 * -np.expm1(-2 / np.pi)
    original = upstream.SpacedSampler.generate_ray_samples
    bundle = rays(16, upstream)
    sampler = upstream.UniformSampler(num_samples=31)
    torch.manual_seed(0)
    baseline = sampler(bundle)
    assert torch.all(baseline.frustums.starts[:, 0] > 0)
    assert torch.all(baseline.frustums.ends[:, -1] < 1)
    with use_method(upstream, "nerfstudio-fix"):
        torch.manual_seed(0)
        corrected = sampler(bundle)
        torch.testing.assert_close(corrected.frustums.starts[:, 0], bundle.nears)
        torch.testing.assert_close(corrected.frustums.ends[:, -1], bundle.fars)
        torch.testing.assert_close(corrected.frustums.starts[:, 1:], baseline.frustums.starts[:, 1:])
        torch.testing.assert_close(corrected.frustums.ends[:, :-1], baseline.frustums.ends[:, :-1])
        torch.testing.assert_close(corrected.deltas.sum(-2), bundle.fars - bundle.nears)
        constant = corrected.get_weights(torch.ones_like(corrected.deltas)).sum(-2)
        torch.testing.assert_close(constant, torch.full_like(constant, -np.expm1(-1.)))
        # Inverse-depth mapping must also remain consistent after correction.
        bundle.nears.fill_(.2)
        disparity = upstream.LinearDisparitySampler(num_samples=31)(bundle)
        torch.testing.assert_close(disparity.frustums.starts[:, 0], bundle.nears)
        torch.testing.assert_close(disparity.frustums.ends[:, -1], bundle.fars)
        errors = []
        for k in (8, 16, 32):
            sampler = upstream.UniformSampler(num_samples=k)
            sampler.eval()
            samples = sampler(rays(1, upstream))
            density = torch.cos((np.pi / 2) * samples.frustums.get_positions()[..., :1])
            errors.append(abs(float(.5 * samples.get_weights(density).sum()) - exact))
        np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4., rtol=.02)
    assert upstream.SpacedSampler.generate_ray_samples is original
    print("NeRFStudio checks passed: endpoint coverage, unchanged interiors, physical deltas, "
          "constant density, disparity mapping, midpoint order")

