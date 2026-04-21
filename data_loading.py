import os
import re
from dataclasses import dataclass as _dc
from typing import Optional, Union
from attr import dataclass
from matplotlib import lines
import numpy as np
from numpy.typing import NDArray
import pandas as pd


@dataclass
class ThreeDObject:
    vertices: NDArray[np.float64]
    faces: pd.DataFrame  # columns: face_id (int), vertex (int)
    normals: Optional[NDArray[np.float64]] = None


def _load_object_stream(object_name: str) -> str:
    object_path = os.path.join("objects", f"{object_name}.obj")
    if not os.path.exists(object_path):
        raise FileNotFoundError(f"Object file not found: {object_path}")
    with open(object_path, "r") as file:
        return file.read()


def _stream_to_3d_object(obj_str: str) -> ThreeDObject:

    face_id = 0
    lines = obj_str.splitlines()
    vertex_rows = []
    face_vertex = []
    face_face_id = []

    print(f"attempting to load object from stream with {len(lines)} lines...")

    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            vertex_rows.append(  # type: ignore
                (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
            )

        elif line.startswith("f "):
            parts = line.split()
            for part in parts[1:]:
                vi = int(part.split("/", 1)[0]) - 1
                face_face_id.append(face_id)  # type: ignore
                face_vertex.append(vi)  # type: ignore
            face_id += 1

    vertices = pd.DataFrame(vertex_rows, columns=["x", "y", "z"])  # type: ignore
    faces = pd.DataFrame(
        {
            "face_id": face_face_id,  # type: ignore
            "vertex": face_vertex,
        }
    )

    print("finished unpacking stream, now converting to 3D object...")
    return ThreeDObject(vertices=np.array(vertices), faces=faces)


def load_object(object_name: str) -> ThreeDObject:
    obj_str = _load_object_stream(object_name)
    return _stream_to_3d_object(obj_str)


@_dc
class Direction:
    theta: float  # azimuth (radians)
    phi: float  # polar angle (radians)

    def to_unit_vector(self) -> NDArray[np.float64]:
        return np.array(
            [
                np.sin(self.phi) * np.cos(self.theta),
                np.sin(self.phi) * np.sin(self.theta),
                np.cos(self.phi),
            ],
            dtype=np.float64,
        )


@_dc
class Rotation:
    direction: "Direction"
    angle_rad: np.float64


@_dc
class Stretch:
    direction: "Direction"
    factor: np.float64


@_dc
class Translation:
    direction: "Direction"
    x_dist: np.float64
    y_dist: np.float64
    z_dist: np.float64


Op = Union[Rotation, Stretch, Translation]


def load_ops_description(path: str) -> str:
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
    return ""


def load_ops(path: str) -> "list[Op]":
    ops: list[Op] = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Missing op type prefix in line: {raw!r}")
            kind, rest = line.split(":", 1)
            kind = kind.strip().lower()

            def _axis() -> Direction:
                m_theta = re.search(r"theta\s*=\s*([\d.eE+\-]+)", rest)
                m_phi = re.search(r"phi\s*=\s*([\d.eE+\-]+)", rest)
                return Direction(
                    theta=float(m_theta.group(1)) if m_theta else 0.0,
                    phi=float(m_phi.group(1)) if m_phi else 0.0,
                )

            if kind == "rot":
                m = re.search(r"angle\s*=\s*([\d.eE+\-]+)", rest)
                angle = np.float64(m.group(1)) if m else np.float64(0.0)
                ops.append(Rotation(direction=_axis(), angle_rad=angle))
            elif kind == "strc":
                m = re.search(r"factor\s*=\s*([\d.eE+\-]+)", rest)
                factor = np.float64(m.group(1)) if m else np.float64(1.0)
                ops.append(Stretch(direction=_axis(), factor=factor))
            elif kind == "trnlt":

                def _val(key: str) -> np.float64:
                    m = re.search(rf"{key}\s*=\s*([\d.eE+\-]+)", rest)
                    return np.float64(m.group(1)) if m else np.float64(0.0)

                ops.append(
                    Translation(
                        direction=Direction(theta=0.0, phi=0.0),
                        x_dist=_val("x"),
                        y_dist=_val("y"),
                        z_dist=_val("z"),
                    )
                )
            else:
                raise ValueError(f"Unknown op type '{kind}' in line: {raw!r}")
    return ops
