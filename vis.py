import os
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from data_loading import ThreeDObject, Op, Rotation, Stretch, Translation


_CAM_FROM = np.array([0.8, 0.5, 1.0], dtype=np.float64)
_CAM_FROM /= np.linalg.norm(_CAM_FROM)
_FWD = -_CAM_FROM  # direction camera looks (into scene)
_RIGHT = np.cross(_FWD, np.array([0.0, 1.0, 0.0]))
_RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_RIGHT, _FWD)
_UP /= np.linalg.norm(_UP)


def _project(verts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthographic projection. Returns (N, 3): [screen_x, screen_y, depth]."""
    return np.column_stack([verts @ _RIGHT, verts @ _UP, verts @ _CAM_FROM])


def normal_to_rgb(normals: NDArray[np.float64]) -> NDArray[np.uint8]:
    safe = np.nan_to_num(normals, nan=0.0)
    rgb = ((safe + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def _normal_to_rgb_float(normals: NDArray[np.float64]) -> NDArray[np.float64]:
    safe = np.nan_to_num(normals, nan=0.0)
    return ((safe + 1.0) / 2.0).clip(0.0, 1.0)


def _draw_normal_key(ax_ops: Any, n_ops: int, x_frac: float, ops_w: float) -> None:
    """Draw an RGB-normal colour key below the ops list."""
    from matplotlib.patches import Rectangle

    y = 0.93 - n_ops * 0.07 - 0.08
    if y < 0.04:
        return

    SW = 0.22 / ops_w  # swatch width as fraction of panel width (≈0.22 in)
    SH = 0.026  # swatch height in axes-fraction
    TX = x_frac + SW + 0.025  # text starts after swatch + small gap
    STEP = 0.058

    def swatch_row(y_top: float, color: tuple, label: str) -> None:
        ax_ops.add_patch(
            Rectangle(
                (x_frac, y_top - SH),
                SW,
                SH,
                facecolor=color,
                edgecolor="none",
                transform=ax_ops.transAxes,
                clip_on=False,
            )
        )
        ax_ops.text(
            TX,
            y_top - SH / 2,
            label,
            transform=ax_ops.transAxes,
            fontsize=8,
            color="white",
            fontfamily="monospace",
            va="center",
            ha="left",
        )

    # Header
    ax_ops.text(
        x_frac,
        y,
        "─── Normal Colour Key ───",
        transform=ax_ops.transAxes,
        fontsize=8.5,
        color="#aaccff",
        fontfamily="monospace",
        fontweight="bold",
        va="top",
        ha="left",
    )
    y -= 0.05

    # View direction
    vx, vy, vz = _CAM_FROM
    ax_ops.text(
        x_frac,
        y,
        f"v = ({vx:.2f}, {vy:.2f}, {vz:.2f})",
        transform=ax_ops.transAxes,
        fontsize=7.5,
        color="#aaaaaa",
        fontfamily="monospace",
        va="top",
        ha="left",
    )
    y -= STEP

    entries: list[tuple[tuple, str]] = [
        (tuple(_normal_to_rgb_float(np.array([1.0, 0.0, 0.0]))), "X+"),
        (tuple(_normal_to_rgb_float(np.array([-1.0, 0.0, 0.0]))), "X-"),
        (tuple(_normal_to_rgb_float(np.array([0.0, 1.0, 0.0]))), "Y+"),
        (tuple(_normal_to_rgb_float(np.array([0.0, -1.0, 0.0]))), "Y-"),
        (tuple(_normal_to_rgb_float(np.array([0.0, 0.0, 1.0]))), "Z+"),
        (tuple(_normal_to_rgb_float(np.array([0.0, 0.0, -1.0]))), "Z-"),
        (tuple(_normal_to_rgb_float(_CAM_FROM)), "Facing camera  "),
        (tuple(_normal_to_rgb_float(-_CAM_FROM)), "Away from camera"),
    ]

    for color, label in entries:
        if y < 0.02:
            break
        swatch_row(y, color, label)  # type: ignore[arg-type]
        y -= STEP


def _op_label(op: Op) -> str:
    if isinstance(op, Rotation):
        d = op.direction
        return f"Rot: angle rads={float(op.angle_rad):.3f}  theta={d.theta:.3f}  phi={d.phi:.3f}"
    if isinstance(op, Stretch):
        d = op.direction
        return f"Stretch: factor={float(op.factor):.3f}  theta={d.theta:.3f}  phi={d.phi:.3f}"
    if isinstance(op, Translation):
        return f"Translate: x={float(op.x_dist):.3f}  y={float(op.y_dist):.3f}  z={float(op.z_dist):.3f}"
    return repr(op)


class Viewer:
    def __init__(self) -> None:
        pass

    def view_obj(
        self,
        obj: "ThreeDObject",
        save_path: Optional[str] = None,
        show: bool = False,
        label: Optional[str] = None,
        ops: Optional[list[Op]] = None,
        description: Optional[str] = None,
    ) -> None:
        print(
            f"Viewing object with {len(obj.vertices)} vertices and "
            f"{len(obj.faces['face_id'].unique())} faces..."
        )
        verts = obj.vertices  # (N, 3)
        proj = _project(verts)  # (N, 3): sx, sy, depth
        sx, sy, depth = proj[:, 0], proj[:, 1], proj[:, 2]

        face_ids = obj.faces["face_id"].to_numpy()
        vertex_indices = obj.faces["vertex"].to_numpy()
        change = np.flatnonzero(np.diff(face_ids)) + 1
        face_groups = np.split(vertex_indices, change)

        polys: list[NDArray] = []
        depths: list[float] = []
        colors: list[NDArray] = []

        for fv in face_groups:
            polys.append(np.column_stack([sx[fv], sy[fv]]))
            depths.append(float(depth[fv].mean()))
            if obj.normals is not None:
                fn = obj.normals[fv].mean(axis=0)
                n = np.linalg.norm(fn)
                fn = fn / n if n > 1e-8 else fn
                colors.append(_normal_to_rgb_float(fn))
            else:
                colors.append(np.array([0.68, 0.85, 0.90]))

        # Painter's algorithm: draw lowest depth (farthest) first
        order = np.argsort(depths)
        polys = [polys[i] for i in order]
        colors = [colors[i] for i in order]

        x_pad = (sx.max() - sx.min()) * 0.06 + 0.1
        y_pad = (sy.max() - sy.min()) * 0.06 + 0.1
        xl, xr = float(sx.min()) - x_pad, float(sx.max()) + x_pad
        yb, yt = float(sy.min()) - y_pad, float(sy.max()) + y_pad

        has_ops = bool(ops)
        IMG_W = 8.0
        FIG_H = 6.0

        if has_ops:

            from matplotlib.gridspec import GridSpec

            op_lines = [_op_label(o) for o in ops]  # type: ignore[arg-type]

            max_chars = max(
                (len(f"{i+1}. {ln}") for i, ln in enumerate(op_lines)), default=20
            )
            CHAR_W_IN = 0.071
            PANEL_PAD = 0.4
            ops_w = max_chars * CHAR_W_IN + PANEL_PAD * 2
            ops_w = max(ops_w, 3.0)

            fig = plt.figure(figsize=(IMG_W + ops_w, FIG_H))
            fig.patch.set_facecolor("#0d1b2a")
            gs = GridSpec(
                1,
                2,
                figure=fig,
                width_ratios=[ops_w, IMG_W],
                wspace=0.0,
                left=0.0,
                right=1.0,
                top=1.0,
                bottom=0.0,
            )
            ax_ops = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            ax_ops.set_facecolor("#0d1b2a")
            ax_ops.axis("off")
            header = f"Ops ({len(op_lines)})"
            ax_ops.text(
                PANEL_PAD / ops_w,
                0.98,
                header,
                transform=ax_ops.transAxes,
                fontsize=10,
                color="#aaccff",
                fontfamily="monospace",
                fontweight="bold",
                va="top",
                ha="left",
            )
            desc_slots = 0
            if description:
                ax_ops.text(
                    PANEL_PAD / ops_w,
                    0.92,
                    f'"{description}"',
                    transform=ax_ops.transAxes,
                    fontsize=8,
                    color="#ffdd88",
                    fontfamily="monospace",
                    fontstyle="italic",
                    va="top",
                    ha="left",
                )
                desc_slots = 1
            for i, line in enumerate(op_lines):
                ax_ops.text(
                    PANEL_PAD / ops_w,
                    0.93 - desc_slots * 0.07 - i * 0.07,
                    f"{i + 1}. {line}",
                    transform=ax_ops.transAxes,
                    fontsize=8.5,
                    color="white",
                    fontfamily="monospace",
                    va="top",
                    ha="left",
                )
            _draw_normal_key(
                ax_ops, len(op_lines) + desc_slots, PANEL_PAD / ops_w, ops_w
            )
        else:
            from matplotlib.gridspec import GridSpec

            PANEL_PAD = 0.4
            ops_w = 3.0
            fig = plt.figure(figsize=(IMG_W + ops_w, FIG_H))
            fig.patch.set_facecolor("#0d1b2a")
            gs = GridSpec(
                1,
                2,
                figure=fig,
                width_ratios=[ops_w, IMG_W],
                wspace=0.0,
                left=0.0,
                right=1.0,
                top=1.0,
                bottom=0.0,
            )
            ax_ops = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            ax_ops.set_facecolor("#0d1b2a")
            ax_ops.axis("off")
            _draw_normal_key(ax_ops, 0, PANEL_PAD / ops_w, ops_w)

        grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1) * np.ones((1, 2))
        ax.imshow(
            grad,
            aspect="auto",
            cmap="Blues",
            extent=[xl, xr, yb, yt],
            origin="lower",
            zorder=0,
        )

        col = PolyCollection(
            polys,
            facecolors=colors,
            linewidths=1,
            zorder=1,
        )
        ax.add_collection(col)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect("equal")
        ax.axis("off")

        cx, cy, cz = verts.mean(axis=0)
        ax.text(
            0.02,
            0.97,
            f"Center  X={cx:.3f}  Y={cy:.3f}  Z={cz:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            fontfamily="monospace",
            va="top",
            ha="left",
            zorder=3,
        )

        if label is not None:
            ax.set_title(label, fontsize=8, family="monospace", loc="left", pad=6)

        if save_path is not None:
            dirpath = os.path.dirname(save_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            fig.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor=fig.get_facecolor(),
            )

        if show:
            plt.show()
        plt.close(fig)
