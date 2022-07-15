from typing import Sequence, Union
import rich
import typer as t
from pathlib import Path
import numpy as np
import transforms3d
import cv2


class Transforms3DUtils(object):
    @classmethod
    def translation(cls, p: Union[np.ndarray, Sequence[float]]):
        T = np.eye(4)
        T[:3, 3] = np.array(p).ravel()
        return T

    @classmethod
    def rot_x(cls, angle: float):
        T = np.eye(4)
        T[:3, :3] = transforms3d.euler.euler2mat(angle, 0.0, 0.0)
        return T

    @classmethod
    def rot_y(cls, angle: float):
        T = np.eye(4)
        T[:3, :3] = transforms3d.euler.euler2mat(0.0, angle, 0.0)
        return T

    @classmethod
    def rot_z(cls, angle: float):
        T = np.eye(4)
        T[:3, :3] = transforms3d.euler.euler2mat(0.0, 0.0, angle)
        return T

    @classmethod
    def rot_euler(cls, a: float, b: float, c: float):
        T = np.eye(4)
        T[:3, :3] = transforms3d.euler.euler2mat(a, b, c)
        return T


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
    start_pose_index: int = t.Option(0, help="start pose index"),
    neural_twin_file: Path = t.Option(..., help="Neural Twin File"),
    orbit: bool = t.Option(False, help="Orbit around the start pose"),
    spp: int = t.Option(1, help="Number of Samples per Pixel used for rendering"),
    ngp_build_folder: Path = t.Option("build", help="Folder in which Instant is built"),
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

    # Selecting mode
    mode = ngp.TestbedMode.Nerf
    testbed = ngp.Testbed(mode)

    if transforms_file.suffix == ".json":
        transforms = json.load(open(transforms_file))
        # Start pose
        frame = transforms["frames"][start_pose_index]
        T = np.array(frame["transform_matrix"])
    elif transforms_file.suffix == ".txt":
        T = np.loadtxt(transforms_file)

    testbed.load_snapshot(str(neural_twin_file))
    testbed.render_mode = ngp.Depth
    testbed.background_color = np.array([120, 144, 156, 255]) / 255.0
    testbed.exposure = 0
    testbed.sun_dir = [0.577, 0.577, 0.577]
    testbed.up_dir = [0.000, 1.000, 0.000]
    testbed.view_dir = [0.000, -0.006, -1.000]
    testbed.look_at = [0.500, 0.500, 0.500]
    testbed.scale = 1.500
    testbed.fov, testbed.dof, testbed.slice_plane_z = 56, 0.000, 0.000
    testbed.autofocus_target = [0.500, 0.500, 0.500]
    testbed.autofocus = False
    testbed.render_aabb = ngp.BoundingBox([0.29, 0.418, 0.349], [0.688, 0.592, 0.571])

    #

    # Camera Parameters
    width, height = 512, 512
    print("SIZE", width, height)
    downsample_while_moving = 1

    translational_velocity = 0.02
    rotational_velocity = 0.02
    navigation_keys_map = {
        # Translation
        "w": Transforms3DUtils.translation([0, 0, -translational_velocity]),
        "s": Transforms3DUtils.translation([0, 0, translational_velocity]),
        "a": Transforms3DUtils.translation([-translational_velocity, 0, 0]),
        "d": Transforms3DUtils.translation([translational_velocity, 0, 0]),
        "q": Transforms3DUtils.translation([0, -translational_velocity, 0]),
        "e": Transforms3DUtils.translation([0, translational_velocity, 0]),
        # rotations
        "y": Transforms3DUtils.rot_z(rotational_velocity),
        "i": Transforms3DUtils.rot_z(-rotational_velocity),
        "h": Transforms3DUtils.rot_y(rotational_velocity),
        "k": Transforms3DUtils.rot_y(-rotational_velocity),
        "j": Transforms3DUtils.rot_x(-rotational_velocity),
        "u": Transforms3DUtils.rot_x(rotational_velocity),
        # Other
        "v": "save_pose",
        "p": "quit",
    }

    while True:

        # Set rendering resolution as camera resolution

        k = cv2.waitKey(0)
        for key, action in navigation_keys_map.items():
            if k == ord(key):
                if isinstance(action, np.ndarray):

                    if not orbit:
                        T = np.dot(T, action)
                    else:
                        T = np.dot(action, T)

                    # downsample if moving
                    break
                elif isinstance(action, str):
                    if action == "save_pose":
                        np.savetxt("/tmp/pose.txt", T)
                        print("Pose saved to:", "/tmp/pose.txt")
                    elif action == "quit":
                        print("Quitting")
                        sys.exit(0)

        # Render current pose
        testbed.set_nerf_camera_matrix(T[:-1, :])

        testbed.render_mode = ngp.Depth
        rendered_image = testbed.render(
            int(height),
            int(width),
            spp,
            True,
        )
        rendered_image = rendered_image[:, :, 2]
        rendered_image = cv2.resize(rendered_image, (width, height))

        print("DEPTH MIN MAX", rendered_image.min(), rendered_image.max())

        rendered_image = cv2.normalize(
            rendered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3
        )

        output_image = cv2.applyColorMap(rendered_image, cv2.COLORMAP_JET)

        # Show
        cv2.imshow("rgb", output_image)

        wr(output_image)


if __name__ == "__main__":
    t.run(navigate_neural_twin)
