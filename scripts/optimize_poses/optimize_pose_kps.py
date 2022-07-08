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
    end_pose_index: int = t.Option(4, help="end pose index"),
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

    correction = np.eye(4)
    correction[2, 3] = -3

    T_start = np.array(start_frame["transform_matrix"])
    T_end = np.array(end_frame["transform_matrix"])

    T_start = T_start @ correction
    T_end = T_end @ correction

    # L = np.eye(4)
    # L[1, 3] = 0.4
    # T_end = T_end @ L

    testbed.load_snapshot(str(neural_twin_file))
    # testbed.exposure = 0
    testbed.background_color = np.array([1, 1, 1, 1])

    # FLow aargsa
    args = {
        "model": "/home/daniele/work/workspace_python/RAFT/models/raft-sintel.pth",
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False,
    }
    args = Box(args)

    # Camera Parameters
    width, height = 384, 384
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
        # x = x * [
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        # ]
        p = x[:3]
        euler = [0, 0, 0]  # x[3:6]  # * 0.01

        # q =  w, x, y, z
        rot = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], "szyx")
        DT = np.eye(4)
        DT[:3, :3] = rot
        DT[:3, 3] = p
        return DT

    def pose_optimizer(x):
        nonlocal rotational_weight, T_end

        # print(x)
        # Render current pose
        DT = convert_to_pose(x)
        current = np.dot(np.linalg.inv(T_end), DT)  # Transforms3DUtils.translation(x)
        current = np.linalg.inv(current)
        # current = np.dot(T_end, DT)
        end_image = render_frame(current, height, width)

        blend = 0.5 * (start_image + end_image)

        stack = np.hstack(
            (
                OUTPUT(start_image),
                OUTPUT(target_image),
                OUTPUT(end_image),
                OUTPUT(blend),
            )
        )

        try:
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            img1 = (start_image * 255).astype(np.uint8)
            img2 = (end_image * 255).astype(np.uint8)
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append([m])

            match_distances = 0.0
            for g in good:
                p1 = np.array(kp1[g[0].queryIdx].pt)
                p2 = np.array(kp2[g[0].trainIdx].pt)
                match_distances += np.linalg.norm(p1 - p2) * (1 / g[0].distance)

            match_distances /= len(good)

            if len(good) == 0:
                return np.inf
            # cv.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(
                img1,
                kp1,
                img2,
                kp2,
                good,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow("matches", img3)
            cv2.imshow("stack", stack)
            cv2.waitKey(1)

            fitness = match_distances
            print("F", fitness)

            return fitness
        except:
            return np.inf
        # return (start_image - end_image).mean()

    rotational_weight = 1
    rotational_bound = rotational_weight * np.pi * 2
    translational_bound = 3

    # Choose optimizer -> https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
    optimizer = ng.optimizers.CMA(
        parametrization=ng.p.Array(init=[0, 0, 0, 0, 0, 0]).set_bounds(
            [
                -translational_bound,
                -translational_bound,
                -translational_bound,
                -rotational_bound,
                -rotational_bound,
                -rotational_bound,
            ],
            [
                translational_bound,
                translational_bound,
                translational_bound,
                rotational_bound,
                rotational_bound,
                rotational_bound,
            ],
            # method="tanh",
            method="bouncing",
        ),
        budget=3000,
    )

    while True:
        cv2.imshow("start", OUTPUT(start_image))
        cv2.waitKey(0)

        recommendation = optimizer.minimize(pose_optimizer)  # best value
        print(recommendation.value)

        DT = convert_to_pose(recommendation.value)
        T_end = T_end @ DT


if __name__ == "__main__":
    t.run(optimize_pose)
