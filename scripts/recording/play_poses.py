from typing import List, Sequence, Union
import rich
import typer as t
from pathlib import Path
import numpy as np
import transforms3d
import cv2
from pipelime.sequences.readers.filesystem import UnderfolderReader

from trajectories import (
    LinearInOut,
    QuadEaseInOut,
    SineEaseInOut,
    TrajectoryInterpolator,
)


class FrameWriter:
    def __init__(self, output_folder: str):
        self.active = False
        self.counter = 0
        if len(output_folder) > 0:
            self.output_folder = Path(output_folder)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            self.active = True

    def __call__(self, frame: np.ndarray):
        if self.active:
            filename = self.output_folder / f"{self.counter:05d}.png"
            rich.print(
                "Frame saved to", filename, frame.min(), frame.max(), frame.dtype
            )
            cv2.imwrite(str(filename), frame)
            self.counter += 1


def navigate_neural_twin(
    transforms_file: Path = t.Option(
        ..., help="transform json file | single pose numpy txt file"
    ),
    replay_dataset: str = t.Option(..., help="raplay underfolder dataset"),
    neural_twin_file: Path = t.Option(..., help="Neural Twin File"),
    spp: int = t.Option(1, help="Number of Samples per Pixel used for rendering"),
    ngp_build_folder: Path = t.Option("build", help="Folder in which Instant is built"),
    fov: float = t.Option(60.0, help="Field of View"),
    resolution: int = t.Option(1024, help="Resolution of the rendered image"),
    output_folder: str = t.Option(
        "", help="Folder in which to save the rendered images"
    ),
):

    import sys

    sys.path.append(str(ngp_build_folder.absolute()))
    import pyngp as ngp  # type: ignore
    import numpy as np
    import json
    import cv2

    wr = FrameWriter(output_folder)
    debug_mode = len(output_folder) == 0

    replay_dataset = UnderfolderReader(replay_dataset)
    print(len(replay_dataset))

    # Selecting mode
    mode = ngp.TestbedMode.Nerf
    testbed = ngp.Testbed(mode)

    poses = [x["pose"] for x in replay_dataset]
    interpolator = TrajectoryInterpolator(
        poses,
        flyby_steps=100,
        waypoint_steps=50,
        easing=QuadEaseInOut(),
    )
    trajectory = interpolator.build_trajectory()

    testbed.load_snapshot(str(neural_twin_file))
    testbed.background_color = [0.000, 0.000, 0.000, 1.000]
    testbed.exposure = 0
    testbed.sun_dir = [0.577, 0.577, 0.577]
    testbed.up_dir = [0.000, 1.000, 0.000]
    testbed.view_dir = [0.000, -0.006, -1.000]
    testbed.look_at = [0.500, 0.500, 0.500]
    testbed.scale = 1.500
    testbed.fov, testbed.dof, testbed.slice_plane_z = fov, 0.000, 0.000
    testbed.autofocus_target = [0.500, 0.500, 0.500]
    testbed.autofocus = False
    testbed.render_aabb = ngp.BoundingBox([0.29, 0.418, 0.349], [0.688, 0.592, 0.571])

    # Camera Parameters
    width, height = resolution, resolution

    while True:

        if len(trajectory) == 0:
            break

        T = trajectory.pop(0)

        # Render current pose
        testbed.set_nerf_camera_matrix(T[:-1, :])
        rendered_image = testbed.render(
            int(height),
            int(width),
            spp,
            True,
        )
        rendered_image = rendered_image[:, :, :3]
        rendered_image = cv2.resize(rendered_image, (width, height))

        # Show
        output_image = (cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR) * 255).astype(
            np.uint8
        )
        if debug_mode:
            cv2.imshow("rgb", output_image)
            cv2.waitKey(1)
        wr(output_image)


if __name__ == "__main__":
    t.run(navigate_neural_twin)
