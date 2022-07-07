from typing import Literal, Optional, Sequence, Union
import pydantic
import typer as t
from pathlib import Path
import numpy as np
import transforms3d
import cv2
from fastapi.applications import FastAPI
from fastapi import WebSocket, FastAPI
import base64
import json
import uvicorn
import threading


def image_to_base64(img: np.ndarray) -> bytes:
    img_buffer = cv2.imencode(".jpg", img)[1]
    return base64.b64encode(img_buffer).decode("utf-8")


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


class Action(pydantic.BaseModel):
    command: str = ""
    transform: Optional[np.ndarray] = None
    transform_type: Literal["world", "relative"] = "world"

    class Config:
        arbitrary_types_allowed = True


def navigate_neural_twin(
    transforms_file: Path = t.Option(..., help="transform json file "),
    start_pose_index: int = t.Option(0, help="start pose index"),
    neural_twin_file: Path = t.Option(..., help="Neural Twin File"),
    debug: bool = t.Option(False, help="Debug mode"),
    spp: int = t.Option(1, help="Number of Samples per Pixel used for rendering"),
    ngp_build_folder: Path = t.Option("build", help="Folder in which Instant is built"),
):

    import sys

    sys.path.append(str(ngp_build_folder.absolute()))
    import pyngp as ngp  # type: ignore

    ########################################
    # Testbed initialization
    ########################################
    mode = ngp.TestbedMode.Nerf
    testbed = ngp.Testbed(mode)
    testbed.load_snapshot(str(neural_twin_file))
    testbed.exposure = 0
    testbed.background_color = np.array([1, 1, 1, 1])

    ########################################
    # Start Pose
    ########################################
    transforms = json.load(open(transforms_file))
    frame = transforms["frames"][start_pose_index]
    T = np.array(frame["transform_matrix"])

    ########################################
    # Camera Parameters
    ########################################
    width, height = 512, 512
    testbed.fov_axis = 0
    testbed.fov = 40

    ########################################
    # Navigation Parameetrs
    ########################################
    rotational_velocity = 0.2
    translational_velocity = 0.2

    navigation_keys_map = {
        "D": Action(
            command="rot_z",
            transform=Transforms3DUtils.rot_z(rotational_velocity),
            transform_type="world",
        ),
        "A": Action(
            command="rot_z",
            transform=Transforms3DUtils.rot_z(-rotational_velocity),
            transform_type="world",
        ),
        "S": Action(
            command="rot_y",
            transform=Transforms3DUtils.rot_y(rotational_velocity),
            transform_type="world",
        ),
        "W": Action(
            command="rot_y",
            transform=Transforms3DUtils.rot_y(-rotational_velocity),
            transform_type="world",
        ),
        "Q": Action(
            command="translate_z",
            transform=Transforms3DUtils.translation([0, 0, translational_velocity]),
            transform_type="relative",
        ),
        "E": Action(
            command="translate_z",
            transform=Transforms3DUtils.translation([0, 0, -translational_velocity]),
            transform_type="relative",
        ),
        "R": Action(command="reset", transform=None),
    }

    ########################################
    # Buffered data
    ########################################
    shared_image = np.zeros((height, width, 3), dtype=np.uint8)
    received_keys = []

    ########################################
    # Web Service
    ########################################

    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        print("started")
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                if len(data) > 0:
                    received_keys.append(data)
                    print("Key Pressed:", received_keys)
                image = image_to_base64(shared_image)
                await websocket.send_bytes(image)
        except Exception as e:
            print(e)
        finally:
            websocket.close()

    def serve():
        uvicorn.run(
            app,
            host="localhost",
            port=8000,
            debug=True,
        )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    ###########################################
    # Rendering Loop
    ###########################################

    while True:

        # Set rendering resolution as camera resolution
        rw, rh = width, height

        try:
            # Pick action based on last received key
            action = navigation_keys_map[received_keys.pop(0)]

            # Textual Actions
            if action.transform is None:
                if action.command == "reset":
                    T = np.array(frame["transform_matrix"])
                else:
                    raise Exception("Unknown action")

            # Geometry Actions
            else:
                dt = action.transform
                if action.transform_type == "world":
                    T = dt @ T
                elif action.transform_type == "relative":
                    T = T @ dt

                # lower the resolution of the image when moving?
                rw = rw // 1
                rh = rh // 1
        except:
            pass

        # Render current pose
        testbed.set_nerf_camera_matrix(T[:-1, :])
        rendered_image = testbed.render(
            int(rh),
            int(rw),
            spp,
            True,
        )
        rendered_image = rendered_image[:, :, :3]
        rendered_image = cv2.resize(rendered_image, (width, height))

        # Show
        output_image = (cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR) * 255).astype(
            np.uint8
        )
        shared_image = output_image.copy()

        # Debug
        if debug:
            cv2.imshow("rgb", output_image)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break


if __name__ == "__main__":
    t.run(navigate_neural_twin)
