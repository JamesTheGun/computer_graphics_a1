from typing import Union
import numpy as np
from numpy.typing import NDArray

from maths import (
    rotate as rotate_fn,
    stretch as stretch_fn,
    transform as transform_fn,
    generate_normals as generate_normals_fn,
)
from data_loading import (
    ThreeDObject,
    Direction,
    Rotation,
    Stretch,
    Translation,
    Op,
    load_ops,
)


def execute_rotation(
    rotation: Rotation, vertices: NDArray[np.float64]
) -> NDArray[np.float64]:
    return rotate_fn(
        vertices=vertices,
        axis=rotation.direction.to_unit_vector(),
        angle=float(rotation.angle_rad),
    )


def execute_stretch(
    stretch: Stretch, vertices: NDArray[np.float64]
) -> NDArray[np.float64]:
    return stretch_fn(
        vertices=vertices,
        factor=stretch.factor,
        axis=stretch.direction.to_unit_vector(),
    )


def execute_translation(
    translation: Translation, vertices: NDArray[np.float64]
) -> NDArray[np.float64]:
    return transform_fn(
        vertices=vertices,
        distances=np.array(
            [translation.x_dist, translation.y_dist, translation.z_dist]
        ),
    )


def execute_ops(
    obj: ThreeDObject,
    ops: list[Op],
) -> ThreeDObject:
    """Apply a list of ops to an object and return a new ThreeDObject with normals."""
    transformed = obj.vertices.copy()
    for op in ops:
        if isinstance(op, Rotation):
            transformed = execute_rotation(op, transformed)
        elif isinstance(op, Stretch):
            transformed = execute_stretch(op, transformed)
        elif isinstance(op, Translation):
            transformed = execute_translation(op, transformed)
    normals = generate_normals_fn(transformed, obj.faces)
    return ThreeDObject(vertices=transformed, faces=obj.faces, normals=normals)
