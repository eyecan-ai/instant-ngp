from typing import Literal, Optional, Sequence, Union
import pydantic
import rich
import typer as t
from pathlib import Path
import numpy as np
import transforms3d
import cv2
from fastapi.applications import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, FastAPI
import cv2
import base64
import numpy as np
from fastapi.responses import HTMLResponse
import albumentations as A
import pathlib

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <canvas id="viewport" width="800" height="600"></canvas>
        <img id="rviewport" width="800" height="600"></img>

        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.binaryType = "arraybuffer";

            ws.onopen = function(e) {
                console.log("[open] Connection established");
            };
            ws.onmessage = function(event) {
                const arrayBuffer = event.data;
                let base_image = new Image();
                base_image.src = 'data:image/jpg;base64,' + arrayBuffer;

                var canvas = document.getElementById('viewport'),
                context = canvas.getContext('2d');  
                base_image.onload = function(){
                    context.drawImage(base_image, 0, 0,800,800);
                }
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }

            setInterval(function() {
                ws.send('')
            }, 10);
        </script>
    </body>
</html>
"""


def image_to_base64(img: np.ndarray) -> bytes:
    """Given a numpy 2D array, returns a JPEG image in base64 format"""

    # using opencv 2, there are others ways
    img_buffer = cv2.imencode(".jpg", img)[1]
    return base64.b64encode(img_buffer).decode("utf-8")


def get_image(volume, index: int):
    image = volume[:, :, index]
    return image_to_base64(image)


def load_image(path: str):
    img = cv2.imread(path)
    t = A.Compose([A.ShiftScaleRotate(p=1)])
    img = t(image=img)["image"]
    return image_to_base64(img)


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
    import uvicorn
    import threading

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
    testbed.exposure = 0
    testbed.background_color = np.array([1, 1, 1, 1])

    # Camera Parameters
    width, height = 512, 512
    print("SIZE", width, height)
    testbed.fov_axis = 0
    testbed.fov = 40
    downsample_while_moving = 1

    rotational_velocity = 0.2
    translational_velocity = 0.2

    # shared image
    shared_image = np.zeros((height, width, 3), dtype=np.uint8)

    app = FastAPI()

    folder = pathlib.Path(__file__).parent.resolve()

    # @app.get("/")
    # async def get():
    #     html = "".join(open(folder / "html" / "index.html").readlines())
    #     return HTMLResponse(html)

    received_keys = []

    navigation_keys_map = {
        "D": {"world": Transforms3DUtils.rot_z(rotational_velocity)},
        "A": {"world": Transforms3DUtils.rot_z(-rotational_velocity)},
        "S": {"world": Transforms3DUtils.rot_y(rotational_velocity)},
        "W": {"world": Transforms3DUtils.rot_y(-rotational_velocity)},
        "R": "reset",
    }

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

    # Add UnderfolderAPI microservice to main app
    # app.include_router(endpoint)

    while True:

        # Set rendering resolution as camera resolution
        rw, rh = width, height

        try:
            action = navigation_keys_map[received_keys.pop(0)]

            if action.transform is None:
                if action.command == "reset":
                    T = np.array(frame["transform_matrix"])
                else:
                    raise Exception("Unknown action")
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
        # cv2.imshow("rgb", output_image)
        # k = cv2.waitKey(1)
        # if k == ord("q"):
        #     break


if __name__ == "__main__":
    t.run(navigate_neural_twin)
