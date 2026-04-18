import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from data_loading import ThreeDObject

ProjectionMode = Literal[
    "planar", "spherical", "cylindrical", "triplanar", "reflective"
]
PlanarAxis = Literal["x", "y", "z"]


def _sample_texture(
    tex: NDArray[np.uint8],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.uint8]:
    h, w = tex.shape[:2]
    u = u % 1.0
    v = v % 1.0
    px = np.nan_to_num((u * (w - 1)).astype(np.float64), nan=0.0)
    py = np.nan_to_num((v * (h - 1)).astype(np.float64), nan=0.0)
    x0 = np.clip(np.floor(px).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(py).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (px - x0)[:, np.newaxis]
    fy = (py - y0)[:, np.newaxis]

    c00 = tex[y0, x0].astype(np.float64)
    c10 = tex[y0, x1].astype(np.float64)
    c01 = tex[y1, x0].astype(np.float64)
    c11 = tex[y1, x1].astype(np.float64)

    result = (
        c00 * (1 - fx) * (1 - fy)
        + c10 * fx * (1 - fy)
        + c01 * (1 - fx) * fy
        + c11 * fx * fy
    )
    return result.clip(0, 255).astype(np.uint8)


def _planar_uv(
    vertices: NDArray[np.float64],
    axis: PlanarAxis,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    a, b = axes[axis]
    coords_a = vertices[:, a]
    coords_b = vertices[:, b]
    u = (coords_a - coords_a.min()) / (coords_a.max() - coords_a.min() + 1e-9)
    v = (coords_b - coords_b.min()) / (coords_b.max() - coords_b.min() + 1e-9)
    return u, v


def _spherical_uv(
    vertices: NDArray[np.float64],
    centre: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    d = vertices - centre
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    u = np.arctan2(d_norm[:, 0], d_norm[:, 2]) / (2 * np.pi) + 0.5
    v = np.arccos(np.clip(d_norm[:, 1], -1.0, 1.0)) / np.pi
    return u, v


def _cylindrical_uv(
    vertices: NDArray[np.float64],
    centre: NDArray[np.float64],
    axis: PlanarAxis = "y",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    d = vertices - centre
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    h = d[:, axis_idx]
    v = (h - h.min()) / (h.max() - h.min() + 1e-9)
    a_idx, b_idx = [i for i in range(3) if i != axis_idx]
    u = np.arctan2(d[:, a_idx], d[:, b_idx]) / (2 * np.pi) + 0.5
    return u, v


def _reflect(
    I: NDArray[np.float64],
    N: NDArray[np.float64],
) -> NDArray[np.float64]:
    I_norm = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-9)
    N_norm = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    dot_prod = np.sum(I_norm * N_norm, axis=1, keepdims=True)
    return I_norm - 2 * dot_prod * N_norm


def _reflective_uv(
    vertices: NDArray[np.float64],
    normals: NDArray[np.float64],
    light_direction: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    light_dir = np.broadcast_to(light_direction, normals.shape)
    reflected = _reflect(light_dir, normals)
    u = np.arctan2(reflected[:, 0], reflected[:, 2]) / (2 * np.pi) + 0.5
    v = np.arccos(np.clip(reflected[:, 1], -1.0, 1.0)) / np.pi
    return u, v


class TextureMapper:
    def __init__(
        self,
        texture: "NDArray[np.uint8] | str",
        mode: ProjectionMode = "planar",
        planar_axis: PlanarAxis = "z",
        tile_u: float = 1.0,
        tile_v: float = 1.0,
        offset_u: float = 0.0,
        offset_v: float = 0.0,
        rotation_deg: float = 0.0,
        triplanar_sharpness: float = 4.0,
        light_direction: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if isinstance(texture, str):
            self.texture: NDArray[np.uint8] = np.array(
                Image.open(texture).convert("RGB")
            )
        else:
            self.texture = texture
        self.mode = mode
        self.planar_axis = planar_axis
        self.tile_u = tile_u
        self.tile_v = tile_v
        self.offset_u = offset_u
        self.offset_v = offset_v
        self.rotation_deg = rotation_deg
        self.triplanar_sharpness = triplanar_sharpness
        if light_direction is None:
            self.light_direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        else:
            self.light_direction = light_direction

    def _apply_uv_transform(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply tiling, offset, and rotation to raw UVs."""
        u = u * self.tile_u + self.offset_u
        v = v * self.tile_v + self.offset_v
        if self.rotation_deg != 0.0:
            rad = np.radians(self.rotation_deg)
            cu, cv = u - 0.5, v - 0.5
            u = cu * np.cos(rad) - cv * np.sin(rad) + 0.5
            v = cu * np.sin(rad) + cv * np.cos(rad) + 0.5
        return u, v

    def map(self, obj: ThreeDObject) -> NDArray[np.uint8]:
        """Return per-vertex RGB colours (N, 3) uint8 for the given object."""
        verts = obj.vertices
        centre = verts.mean(axis=0)

        if self.mode == "planar":
            u, v = _planar_uv(verts, self.planar_axis)
            u, v = self._apply_uv_transform(u, v)
            return _sample_texture(self.texture, u, v)

        elif self.mode == "spherical":
            u, v = _spherical_uv(verts, centre)
            u, v = self._apply_uv_transform(u, v)
            return _sample_texture(self.texture, u, v)

        elif self.mode == "cylindrical":
            u, v = _cylindrical_uv(verts, centre, self.planar_axis)
            u, v = self._apply_uv_transform(u, v)
            return _sample_texture(self.texture, u, v)

        elif self.mode == "triplanar":
            assert (
                obj.normals is not None
            ), "triplanar mode requires ThreeDObject.normals"
            weights = np.abs(obj.normals) ** self.triplanar_sharpness
            weights /= weights.sum(axis=1, keepdims=True) + 1e-9

            rgb = np.zeros((len(verts), 3), dtype=np.float64)
            for i, axis in enumerate(("x", "y", "z")):
                u, v = _planar_uv(verts, axis)
                u, v = self._apply_uv_transform(u, v)
                sampled = _sample_texture(self.texture, u, v).astype(np.float64)
                rgb += sampled * weights[:, i : i + 1]

            return rgb.clip(0, 255).astype(np.uint8)

        elif self.mode == "reflective":
            assert (
                obj.normals is not None
            ), "reflective mode requires ThreeDObject.normals"
            u, v = _reflective_uv(verts, obj.normals, self.light_direction)
            u, v = self._apply_uv_transform(u, v)
            return _sample_texture(self.texture, u, v)

        else:
            raise ValueError(f"Unknown projection mode: {self.mode!r}")


COMBINE_OPTIONS = [
    "multiply",
    "add",
    "screen",
    "overlay",
    "normal_warp",
    "xor_hue",
    "lightness",
    "inverse_lightness",
]

CombineMode = Literal[
    "multiply",
    "add",
    "screen",
    "overlay",
    "normal_warp",
    "xor_hue",
    "lightness",
    "inverse_lightness",
]


def combine_normals(
    texture: NDArray[np.float64],
    normals: NDArray[np.float64],
    mode: CombineMode = "multiply",
) -> NDArray[np.float64]:
    t = np.clip((texture + 1.0) / 2.0, 0.0, 1.0)
    n_raw = (normals + 1.0) / 2.0
    n_min = n_raw.min(axis=0, keepdims=True)
    n_max = n_raw.max(axis=0, keepdims=True)
    n = np.clip((n_raw - n_min) / (n_max - n_min + 1e-9), 0.0, 1.0)

    if mode == "multiply":
        out = t * n

    elif mode == "add":
        out = np.clip(t * 0.7 + (t + n) * 0.3 - 0.15, 0.0, 1.0)

    elif mode == "screen":
        screen = 1.0 - (1.0 - t) * (1.0 - n)
        out = t * 0.5 + screen * 0.5

    elif mode == "overlay":
        out = (1.0 - 2.0 * n) * t**2 + 2.0 * n * t

    elif mode == "normal_warp":
        shift = (n - 0.5) * 0.3
        out = np.clip(t + shift, 0.0, 1.0)

    elif mode == "xor_hue":
        theta = np.arctan2(normals[:, 0], normals[:, 2]) / (2.0 * np.pi) + 0.5
        shifted = (t + theta[:, np.newaxis]) % 1.0
        luma_t = 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]
        out = shifted * 0.6 + luma_t * 0.4

    elif mode == "lightness":
        luma = np.clip((normals[:, 1] + 1.0) / 2.0, 0.0, 1.0)
        luma = (luma - luma.min()) / (luma.max() - luma.min() + 1e-9)
        brightness = (luma - 0.5) * 0.8
        out = np.clip(t + brightness[:, np.newaxis], 0.0, 1.0)

    elif mode == "inverse_lightness":
        luma = np.clip((normals[:, 1] + 1.0) / 2.0, 0.0, 1.0)
        luma = (luma - luma.min()) / (luma.max() - luma.min() + 1e-9)
        brightness = (0.5 - luma) * 0.8
        out = np.clip(t + brightness[:, np.newaxis], 0.0, 1.0)

    else:
        raise ValueError(f"Unknown combine mode: {mode!r}")

    return out * 2.0 - 1.0


def apply_texture_to_obj(
    obj: ThreeDObject,
    texture: "NDArray[np.uint8] | str",
    mode: ProjectionMode = "planar",
    planar_axis: PlanarAxis = "z",
    tile_u: float = 1.0,
    tile_v: float = 1.0,
    offset_u: float = 0.0,
    offset_v: float = 0.0,
    rotation_deg: float = 0.0,
    triplanar_sharpness: float = 4.0,
    combine_mode: CombineMode = "multiply",
) -> ThreeDObject:
    mapper = TextureMapper(
        texture=texture,
        mode=mode,
        planar_axis=planar_axis,
        tile_u=tile_u,
        tile_v=tile_v,
        offset_u=offset_u,
        offset_v=offset_v,
        rotation_deg=rotation_deg,
        triplanar_sharpness=triplanar_sharpness,
    )
    rgb_uint8 = mapper.map(obj)
    rgb_as_normals = (rgb_uint8.astype(np.float64) / 255.0) * 2.0 - 1.0
    if obj.normals is not None:
        rgb_as_normals = combine_normals(rgb_as_normals, obj.normals, mode=combine_mode)
    return ThreeDObject(vertices=obj.vertices, faces=obj.faces, normals=rgb_as_normals)
