import os
from typing import Optional

import pyvista as pv
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from data_loading import ThreeDObject


def normal_to_intensity(
    normals: NDArray[np.float64],
    light_angle: float = 0.0,
) -> NDArray[np.float64]:
    light_dir = np.array([np.sin(light_angle), 0.0, np.cos(light_angle)])
    raw = np.dot(normals, light_dir)
    i_min, i_max = float(raw.min()), float(raw.max())
    if i_max > i_min:
        return (raw - i_min) / (i_max - i_min)
    return np.full_like(raw, 0.5)


def normal_to_rgb(normals: NDArray[np.float64]) -> NDArray[np.uint8]:
    safe = np.nan_to_num(normals, nan=0.0)
    rgb = ((safe + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


class Viewer:
    def __init__(self) -> None:
        """Initialize the Viewer."""
        self.plotter: pv.Plotter = pv.Plotter()

    def hist_normals(self, obj: ThreeDObject, light_angle: float = 0.0) -> None:
        assert obj.normals is not None, "Object must have normals to plot histogram"
        rgb = normal_to_rgb(obj.normals)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axis_labels = ["X  ->  Red", "Y  ->  Green", "Z  ->  Blue"]
        colours = ["#d9534f", "#5cb85c", "#337ab7"]

        for ax, col_idx, xlabel, colour in zip(axes, range(3), axis_labels, colours):
            data = obj.normals[:, col_idx]
            n, bins, patches = ax.hist(data, bins=40, color=colour, alpha=0.85)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            channel_vals = ((bin_centers + 1.0) / 2.0).clip(0, 1)
            base = np.zeros(3)
            for patch, v in zip(patches, channel_vals):
                c = base.copy()
                c[col_idx] = v
                patch.set_facecolor(c)
            ax.set_title(xlabel)
            ax.set_xlabel("Normal value")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Normal distribution: XYZ -> RGB", fontsize=11)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def view_obj(
        self,
        obj: "ThreeDObject",
        show_edges: bool = False,
        camera_dir: Optional[NDArray[np.float64]] = None,
        light_angle: float = 0.0,
        save_path: Optional[str] = None,
        show: bool = False,
        label: Optional[str] = None,
    ) -> None:
        print(
            f"Viewing object with {len(obj.vertices)} vertices and {len(obj.faces['face_id'].unique())} faces..."
        )
        self.plotter = pv.Plotter(off_screen=True)

        face_ids = obj.faces["face_id"].to_numpy()
        vertices = obj.faces["vertex"].to_numpy()

        change = np.flatnonzero(np.diff(face_ids)) + 1
        face_groups = np.split(vertices, change)

        formatted_faces = np.hstack([[len(f)] + list(f) for f in face_groups])

        mesh = pv.PolyData(obj.vertices.astype(np.float32), formatted_faces)

        if obj.normals is not None:
            if camera_dir is not None:
                cam_dir_norm = camera_dir / np.linalg.norm(camera_dir)
                self.plotter.camera.position = -cam_dir_norm * 10  # type: ignore
                self.plotter.reset_camera()  # type: ignore
            else:
                self.plotter.add_mesh(mesh, color="lightblue", show_edges=show_edges)  # type: ignore
                self.plotter.reset_camera()  # type: ignore
                cam_pos = np.array(self.plotter.camera.position)  # type: ignore
                focal = np.array(self.plotter.camera.focal_point)  # type: ignore
                cam_dir_norm = focal - cam_pos
                cam_dir_norm = cam_dir_norm / np.linalg.norm(cam_dir_norm)

            rgb = normal_to_rgb(obj.normals)

            self.plotter.clear()
            mesh["RGB"] = rgb
            self.plotter.add_mesh(mesh, scalars="RGB", rgb=True, show_edges=show_edges)  # type: ignore
            self.plotter.reset_camera()  # type: ignore
        else:
            self.plotter.add_mesh(mesh, color="lightblue", show_edges=show_edges)  # type: ignore
            self.plotter.reset_camera()  # type: ignore

        img: Optional[NDArray] = self.plotter.screenshot(return_img=True)  # type: ignore
        assert img is not None, "Screenshot failed"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.axis("off")
        if label is not None:
            ax.set_title(label, fontsize=8, family="monospace", loc="left", pad=6)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        plt.close(fig)
