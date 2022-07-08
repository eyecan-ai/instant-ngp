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


class Renderer:
    def __init__(
        self, twin_file: str, rendering_size: Sequence[int] = [512, 512], spp: int = 1
    ) -> None:
        # Selecting mode
        import pyngp as ngp  # type: ignore

        mode = ngp.TestbedMode.Nerf
        self._spp = spp
        self._testbed = ngp.Testbed(mode)
        self._testbed.load_snapshot(twin_file)
        self._testbed.exposure = 0
        self._testbed.background_color = np.array([1, 1, 1, 1])

        # Camera Parameters
        self.width, self.height = rendering_size
        self._testbed.fov_axis = 0
        self._testbed.fov = 40

    def set_pose(self, pose: np.ndarray):
        self._testbed.set_nerf_camera_matrix(pose[:-1, :])

    def render(self) -> np.ndarray:
        return self._testbed.render(
            int(self.height),
            int(self.width),
            self._spp,
            True,
        )


def navigate_neural_twin(
    transforms_file: Path = t.Option(
        ..., help="transform json file | single pose numpy txt file"
    ),
    start_pose_index: int = t.Option(0, help="start pose index"),
    first_twin: Path = t.Option(..., help="First Neural Twin File"),
    second_twin: Path = t.Option(..., help="Second Neural Twin File"),
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
    renderer_first = Renderer(str(first_twin), [512, 512], spp=spp)
    renderer_second = Renderer(str(second_twin), [512, 512], spp=spp)

    if transforms_file.suffix == ".json":
        transforms = json.load(open(transforms_file))
        # Start pose
        frame = transforms["frames"][start_pose_index]
        T = np.array(frame["transform_matrix"])
    elif transforms_file.suffix == ".txt":
        T = np.loadtxt(transforms_file)

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

    OUT = lambda x: (cv2.cvtColor(x, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)

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
        renderer_first.set_pose(T)
        renderer_second.set_pose(T)

        rendered_image_first = renderer_first.render()[:, :, :3]
        rendered_image_second = renderer_second.render()[:, :, :3]

        difference = np.abs(rendered_image_first - rendered_image_second)

        # Show
        stack = np.hstack(
            (
                OUT(rendered_image_first),
                OUT(rendered_image_second),
                OUT(difference),
            )
        )

        cv2.imshow("multiview", stack)
        # wr(output_image)


if __name__ == "__main__":
    t.run(navigate_neural_twin)
