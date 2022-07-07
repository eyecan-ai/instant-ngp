from typing import Sequence, Union
import rich
import typer as t
from pathlib import Path
import numpy as np
import transforms3d
import cv2
import nevergrad as ng
import sys

sys.path.append("core")

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from box import Box

sys.path.append("/home/daniele/work/workspace_python/RAFT/core")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from types import SimpleNamespace

DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(flo, name="image"):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo[:, :, [2, 1, 0]] / 255.0


def convert_image(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def compute_flow(flow_model, image1, image2, iters=20):

    image1 = convert_image(image1)
    image2 = convert_image(image2)
    with torch.no_grad():

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = flow_model(image1, image2, iters=iters, test_mode=True)

        return flow_up


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


def optimize_pose(
    transforms_file: Path = t.Option(..., help="transform json file "),
    start_pose_index: int = t.Option(0, help="start pose index"),
    end_pose_index: int = t.Option(10, help="end pose index"),
    neural_twin_file: Path = t.Option(..., help="Neural Twin File"),
    spp: int = t.Option(1, help="Number of Samples per Pixel used for rendering"),
    ngp_build_folder: Path = t.Option("build", help="Folder in which Instant is built"),
):
    import sys

    sys.path.append(str(ngp_build_folder.absolute()))

    import pyngp as ngp  # type: ignore
    import numpy as np
    import json
    import cv2

    # Selecting mode
    mode = ngp.TestbedMode.Nerf
    testbed = ngp.Testbed(mode)

    transforms = json.load(open(transforms_file))

    # Start pose
    start_frame = transforms["frames"][start_pose_index]
    end_frame = transforms["frames"][end_pose_index]
    print("Loading poses", start_pose_index, end_pose_index)

    T_start = np.array(start_frame["transform_matrix"])
    T_end = np.array(end_frame["transform_matrix"])

    L = np.eye(4)
    L[1, 3] = 0.4
    T_end = T_end @ L

    testbed.load_snapshot(str(neural_twin_file))
    # testbed.exposure = 0
    # testbed.background_color = np.array([0, 0, 0, 0])

    # FLow aargsa
    args = {
        "model": "/home/daniele/work/workspace_python/RAFT/models/raft-sintel.pth",
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False,
    }
    args = Box(args)
    flow_model = torch.nn.DataParallel(RAFT(args))
    flow_model.load_state_dict(torch.load(args.model))

    flow_model = flow_model.module
    flow_model.to(DEVICE)
    flow_model.eval()

    # Camera Parameters
    width, height = 256, 256
    print("SIZE", width, height)
    testbed.fov_axis = 0
    testbed.fov = 40

    def render_frame(T: np.ndarray, height: int, width: int):
        testbed.set_nerf_camera_matrix(T[:-1, :])
        return testbed.render(
            int(height),
            int(width),
            spp,
            True,
        )[:, :, :3]

    start_image = render_frame(T_start, height, width)
    target_image = render_frame(T_end, height, width)

    OUTPUT = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imshow("start", OUTPUT(start_image))
    cv2.imshow("target", OUTPUT(target_image))
    cv2.waitKey(0)

    def convert_to_pose(x):
        x = x * [
            1,
            1,
            1,
            1 / 10,
            1 / 10,
            1 / 10,
        ]
        p = x[:3]
        euler = x[3:6]  # * 0.01

        # q =  w, x, y, z
        rot = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
        DT = np.eye(4)
        DT[:3, :3] = rot
        DT[:3, 3] = p
        return DT

    def pose_optimizer(x):
        nonlocal rotational_weight, T_end

        # print(x)
        # Render current pose
        DT = convert_to_pose(x)
        current = np.dot(T_end, DT)  # Transforms3DUtils.translation(x)

        end_image = render_frame(current, height, width)

        flow = compute_flow(
            flow_model,
            start_image * 255,
            end_image * 255.0,
            iters=5,
        )
        # flow_BA = compute_flow(flow_model, end_image * 255, start_image * 255)

        flow_color = viz(flow, "ab")
        # viz(flow_BA, "ba")

        blend = 0.5 * (start_image + end_image)

        stack = np.hstack(
            (
                OUTPUT(start_image),
                OUTPUT(target_image),
                OUTPUT(end_image),
                OUTPUT(blend),
                flow_color,
            )
        )

        cv2.imshow("stack", stack)
        cv2.waitKey(1)

        # flow_mag = torch.norm(flow, dim=1).mean()
        flow_mag = torch.abs(flow).mean()
        # flow_mag = torch.abs(flow).mean()
        # mag_BA = torch.norm(flow_BA, dim=1).mean()

        variance = 1.0 - end_image.var()
        fitness = (flow_mag).item() + 1000 * variance
        # fitness = 255 * np.abs(start_image - end_image).mean()

        print("F", fitness)

        return fitness
        # return (start_image - end_image).mean()

    rotational_weight = 1
    rotational_bound = rotational_weight * np.pi * 5
    translational_bound = 5

    optimizer = ng.optimizers.NGOpt(
        parametrization=ng.p.Array(init=[0, 0, 0, 0, 0, 0]).set_bounds(
            [-translational_bound] * 3 + [-rotational_bound] * 3,
            [translational_bound] * 3 + [rotational_bound] * 3,
            # method="tanh",
            method="bouncing",
        ),
        budget=300,
    )

    while True:
        cv2.imshow("start", OUTPUT(start_image))
        cv2.waitKey(0)

        # Choose optimizer -> https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer

        recommendation = optimizer.minimize(pose_optimizer)  # best value
        print(recommendation.value)

        DT = convert_to_pose(recommendation.value)
        T_end = T_end @ DT


if __name__ == "__main__":
    t.run(optimize_pose)
