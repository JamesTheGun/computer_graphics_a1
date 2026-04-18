import os
import shutil
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from texture_stuff.mapping import apply_texture_to_obj, COMBINE_OPTIONS
from data_loading import ThreeDObject, Op, load_ops, load_object
from utils import execute_ops
from vis import Viewer

DEFAULT_OPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops.txt")


class FuckWithObject:
    def __init__(
        self,
        obj_path: str,
        output_dir: str = "outputs",
        show_edges: bool = False,
        ops_path: str = DEFAULT_OPS_PATH,
        texture_path: Optional[str] = None,
        texture_mode: str = "triplanar",
        texture_tile_u: float = 1.0,
        texture_tile_v: float = 1.0,
    ) -> None:
        self.obj_path = obj_path
        ops_stem = os.path.splitext(os.path.basename(ops_path))[0]
        self.output_dir = os.path.join(output_dir, os.path.basename(obj_path), ops_stem)
        self.ops_path = ops_path
        self.show_edges = show_edges
        self.texture_path = texture_path
        self.texture_mode = texture_mode
        self.texture_tile_u = texture_tile_u
        self.texture_tile_v = texture_tile_v
        self._viewer = Viewer()

    def _load(self) -> ThreeDObject:
        return load_object(self.obj_path)

    def _clear_output_dir(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir)

    def run(self) -> None:
        self._clear_output_dir()
        obj = self._load()
        ops: list[Op] = load_ops(self.ops_path)
        result_obj = execute_ops(obj, ops)

        # Save the plain (no texture) render
        self._viewer.view_obj(
            result_obj,
            save_path=os.path.join(self.output_dir, "result.png"),
            show_edges=self.show_edges,
        )

        if self.texture_path is not None:
            for combine_mode in COMBINE_OPTIONS:
                display_obj = apply_texture_to_obj(
                    result_obj,
                    self.texture_path,
                    mode=self.texture_mode,  # type: ignore[arg-type]
                    tile_u=self.texture_tile_u,
                    tile_v=self.texture_tile_v,
                    combine_mode=combine_mode,  # type: ignore[arg-type]
                )
                save_path = os.path.join(self.output_dir, f"{combine_mode}.png")
                self._viewer.view_obj(
                    display_obj,
                    save_path=save_path,
                    show_edges=self.show_edges,
                )


if __name__ == "__main__":
    for texture_mode in ["triplanar", "planar", "reflective"]:
        for obj_name in ["cow", "crock", "mystery", "monkey"]:
            for ops in ["ops.txt", "ops2.txt", "ops3.txt", "ops4.txt"]:
                striped_ops = os.path.splitext(ops)[0]
                FuckWithObject(
                    os.path.join(obj_name),
                    texture_path=f"C:\\Users\\jamed\\Desktop\\computer_graphics_a1\\textures\\checkerboard.png",
                    ops_path=ops,
                    output_dir=os.path.join(f"outputs", texture_mode),
                    texture_mode=texture_mode,
                ).run()
