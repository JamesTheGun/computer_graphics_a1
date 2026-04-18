import os
import shutil
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from PIL import Image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures")


def _save(img_array: NDArray[np.uint8], name: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    Image.fromarray(img_array).save(path)
    return path


def checkerboard(
    width: int = 512,
    height: int = 512,
    tile_size: int = 64,
    colour_a: tuple[int, int, int] = (255, 255, 255),
    colour_b: tuple[int, int, int] = (30, 30, 30),
) -> str:
    """Alternating solid-colour tiles."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                img[y, x] = colour_a
            else:
                img[y, x] = colour_b
    return _save(img, "checkerboard.png")


def noise(
    width: int = 512,
    height: int = 512,
    scale: float = 1.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> str:
    """Fractal Brownian Motion (fBm) grayscale noise texture."""
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    frequency = scale / min(width, height)
    for _ in range(octaves):
        # Random gradient noise via sinusoidal hash
        xs = np.linspace(0, width * frequency, width, endpoint=False)
        ys = np.linspace(0, height * frequency, height, endpoint=False)
        gx, gy = np.meshgrid(xs, ys)
        phase = rng.uniform(0, 2 * np.pi, size=(2,))
        layer = np.sin(gx * 2 * np.pi + phase[0]) * np.cos(gy * 2 * np.pi + phase[1])
        result += amplitude * layer
        amplitude *= persistence
        frequency *= lacunarity
    # Normalise to [0, 255]
    result = (result - result.min()) / (result.max() - result.min())
    img = (result * 255).astype(np.uint8)
    img_rgb = np.stack([img, img, img], axis=-1)
    return _save(img_rgb, "noise.png")


def gradient(
    width: int = 512,
    height: int = 512,
    colour_start: tuple[int, int, int] = (255, 0, 128),
    colour_end: tuple[int, int, int] = (0, 128, 255),
    direction: str = "horizontal",
) -> str:
    """Linear gradient between two colours. direction: 'horizontal' | 'vertical' | 'diagonal'."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if direction == "horizontal":
        t = np.linspace(0, 1, width)
        img[:] = (
            np.array(colour_start) * (1 - t[:, None])
            + np.array(colour_end) * t[:, None]
        ).astype(np.uint8)
    elif direction == "vertical":
        t = np.linspace(0, 1, height)
        img[:] = (
            np.array(colour_start) * (1 - t[:, None])
            + np.array(colour_end) * t[:, None]
        ).astype(np.uint8)[:, np.newaxis, :]
    else:  # diagonal
        tx = np.linspace(0, 1, width)
        ty = np.linspace(0, 1, height)
        t = np.clip((tx[np.newaxis, :] + ty[:, np.newaxis]) / 2, 0, 1)
        img = (
            np.array(colour_start) * (1 - t[:, :, np.newaxis])
            + np.array(colour_end) * t[:, :, np.newaxis]
        ).astype(np.uint8)
    return _save(img, f"gradient_{direction}.png")


def voronoi(
    width: int = 512,
    height: int = 512,
    num_cells: int = 32,
    seed: int = 0,
    coloured: bool = True,
) -> str:
    """Voronoi / cellular texture."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, size=(num_cells, 2))
    cell_colours = rng.integers(40, 255, size=(num_cells, 3), dtype=np.uint8)

    xs = np.linspace(0, 1, width)
    ys = np.linspace(0, 1, height)
    gx, gy = np.meshgrid(xs, ys)
    px = gx[:, :, np.newaxis]  # (H, W, 1)
    py = gy[:, :, np.newaxis]
    dist = np.sqrt((px - points[:, 0]) ** 2 + (py - points[:, 1]) ** 2)
    nearest = np.argmin(dist, axis=2)  # (H, W)

    if coloured:
        img = cell_colours[nearest]
    else:
        # Distance to nearest point as grayscale
        min_dist = dist[np.arange(height)[:, None], np.arange(width)[None, :], nearest]
        norm = (min_dist / min_dist.max() * 255).astype(np.uint8)
        img = np.stack([norm, norm, norm], axis=-1)

    return _save(img, "voronoi.png")


def stripes(
    width: int = 512,
    height: int = 512,
    stripe_width: int = 32,
    angle_deg: float = 45.0,
    colour_a: tuple[int, int, int] = (220, 50, 50),
    colour_b: tuple[int, int, int] = (240, 240, 240),
) -> str:
    """Angled stripes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    angle = np.radians(angle_deg)
    xs = np.arange(width)
    ys = np.arange(height)
    gx, gy = np.meshgrid(xs, ys)
    proj = (gx * np.cos(angle) + gy * np.sin(angle)).astype(int)
    mask = (proj // stripe_width) % 2 == 0
    img[mask] = colour_a
    img[~mask] = colour_b
    return _save(img, "stripes.png")


class TextureGenerator:
    """Convenience wrapper — call .run() to generate all textures."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        seed: int = 42,
    ) -> None:
        global OUTPUT_DIR
        if output_dir is not None:
            OUTPUT_DIR = output_dir
        self.width = width
        self.height = height
        self.seed = seed

    def run(self) -> None:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR)

        paths = [
            checkerboard(self.width, self.height),
            noise(self.width, self.height, seed=self.seed),
            gradient(self.width, self.height, direction="horizontal"),
            gradient(self.width, self.height, direction="vertical"),
            gradient(self.width, self.height, direction="diagonal"),
            voronoi(self.width, self.height, seed=self.seed),
            stripes(self.width, self.height),
        ]
        for p in paths:
            print(f"Saved {p}")


if __name__ == "__main__":
    TextureGenerator().run()
