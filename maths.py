import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional


def rotate(
    vertices: NDArray[np.float64],
    axis: NDArray[np.float64],
    angle: float,
    centroid: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    if centroid is None:
        centroid = np.median(vertices, axis=0)

    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0 or angle == 0.0:
        return vertices.copy()

    axis = axis / axis_norm

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    x, y, z = axis

    K = np.array(  # type: ignore
        [  # type: ignore
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )

    rodrigues = (   # type: ignore
        lambda c, s, K, u: np.eye(3) * c + (1.0 - c) * np.outer(u, u) + s * K   # type: ignore
    )

    rod_rotation_matrix = rodrigues(cos_theta, sin_theta, K, axis)   # type: ignore

    centered = vertices - centroid
    rotated = centered @ rod_rotation_matrix.T   # type: ignore
    return rotated + centroid   # type: ignore


def stretch(
    vertices: NDArray[np.float64],
    factor: float,
    axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert axis.shape == (3,)

    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        return vertices.copy()

    axis = axis / axis_norm

    scalars = vertices @ axis
    parallel = scalars[:, None] * axis
    perpendicular = vertices - parallel

    return perpendicular + factor * parallel


def uniform_scale(
    vertices: NDArray[np.float64],
    factor: float,
) -> NDArray[np.float64]:
    vertices = stretch(vertices, factor, np.array([1.0, 0.0, 0.0]))
    vertices = stretch(vertices, factor, np.array([0.0, 1.0, 0.0]))
    vertices = stretch(vertices, factor, np.array([0.0, 0.0, 1.0]))
    return vertices


def _faces_to_triangles(faces: pd.DataFrame) -> NDArray[np.intp]:
    faces = faces.sort_values(["face_id"]).reset_index(drop=True)

    verts = faces["vertex"].to_numpy(np.intp)
    face_ids = faces["face_id"].to_numpy()

    change = np.flatnonzero(np.diff(face_ids)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(verts)]))

    counts = ends - starts - 2
    valid = counts > 0

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.intp)

    starts = starts[valid]
    counts = counts[valid]

    v0 = np.repeat(verts[starts], counts)

    offsets = np.concatenate([np.arange(c) for c in counts])
    base = np.repeat(starts, counts)

    v1 = verts[base + offsets + 1]
    v2 = verts[base + offsets + 2]
    return np.stack((v0, v1, v2), axis=1)


def generate_normals(
    vertices: NDArray[np.float64], faces: pd.DataFrame
) -> NDArray[np.float64]:

    vertices = np.asarray(vertices, dtype=np.float64)
    tris = _faces_to_triangles(faces)

    if len(tris) == 0:
        return np.zeros_like(vertices)

    tri = vertices[tris]

    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]

    face_normals = np.cross(edge1, edge2)

    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    nonzero = lengths[:, 0] > 0
    face_normals[nonzero] /= lengths[nonzero]

    normals = np.zeros_like(vertices)
    np.add.at(normals, tris[:, 0], face_normals)
    np.add.at(normals, tris[:, 1], face_normals)
    np.add.at(normals, tris[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    nonzero = lengths[:, 0] > 0
    normals[nonzero] /= lengths[nonzero]
    return normals


def transform(
    vertices: NDArray[np.float64], distances: NDArray[np.float64]
) -> NDArray[np.float64]:
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert distances.shape == (3,)
    return vertices + distances
