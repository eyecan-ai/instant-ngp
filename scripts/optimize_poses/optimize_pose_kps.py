from typing import Sequence, Union
import rich
import typer as t
from pathlib import Path
import numpy as np
import transforms3d
import cv2
import nevergrad as ng
import sys
import albumentations as A

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
import imageio

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


class ImagePairsLoss:
    def __init__(self) -> None:
        pass

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> float:
        raise NotImplementedError("ImagePairsLoss is not implemented")


class FeaturesImageLoss:
    def __init__(
        self,
        algo: str = "sift",
        expected_max_matches: int = 100,
        max_matches_weights: float = 10,
        debug: bool = True,
    ) -> None:

        self._algo = algo
        self._expected_max_matches = expected_max_matches
        self._max_matches_weights = max_matches_weights
        self._debug = debug

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> float:

        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)

        if self._algo == "sift":

            sift = cv2.SIFT_create()

            # detect / describe
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # Compute matches
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            max_distance = 0.0
            for m, n in matches:
                if m.distance > max_distance:
                    max_distance = m.distance
                if n.distance > max_distance:
                    max_distance = n.distance
                if m.distance < 0.6 * n.distance:
                    good.append([m])

            # Compute average keypoints distance loss
            loss = 0.0
            for g in good:
                p1 = np.array(kp1[g[0].queryIdx].pt)
                p2 = np.array(kp2[g[0].trainIdx].pt)
                norm_distance = max_distance / g[0].distance

                loss += (np.linalg.norm(p1 - p2) * (1 / norm_distance)) ** 2

            loss /= len(good)  # normalize distance loss on number of matches

            # increase loss proportional to the inverse of number of matches
            # More matches -> Lower loss
            loss += self._max_matches_weights * max(
                (self._expected_max_matches - len(good)) / self._expected_max_matches, 0
            )

            if self._debug:
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
                cv2.waitKey(1)

            return loss
        else:
            raise NotImplementedError("Algo {} is not implemented".format(self._algo))


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


def optimize_pose(
    transforms_file: Path = t.Option(..., help="transform json file "),
    start_pose_index: int = t.Option(12, help="start pose index"),
    end_pose_index: int = t.Option(22, help="end pose index"),
    neural_twin_file: Path = t.Option(..., help="Neural Twin File"),
    spp: int = t.Option(1, help="Number of Samples per Pixel used for rendering"),
    ngp_build_folder: Path = t.Option("build", help="Folder in which Instant is built"),
    fov: float = t.Option(40.0, help="Field of View"),
    output_folder: str = t.Option("", help="Output folder"),
    epochs: int = t.Option(3000, help="Number of epochs"),
    warmap_epochs: int = t.Option(200, help="Number of epochs for warmap"),
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
    width, height = 256, 256
    print("SIZE", width, height)
    testbed.fov_axis = 0
    testbed.fov = fov
    # testbed.exposure = 1

    def render_frame(T: np.ndarray, height: int, width: int):
        testbed.set_nerf_camera_matrix(T[:-1, :])
        return testbed.render(
            int(height),
            int(width),
            spp,
            True,
        )[:, :, :3]

    start_image = render_frame(T_start, height, width)

    # LOAD START IMAGE FROM FILE
    start_image = (
        imageio.imread(
            "/home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/datasets/lucas_candles/nerf/images/00012_marker_0010.png"
        )[:, :, :3]
        / 255.0
    ).astype(np.float32)

    t = A.Compose([A.CenterCrop(500, 500), A.Resize(width, height)])
    start_image = t(image=start_image)["image"]

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
        euler = x[3:6]  # * 0.01

        # q =  w, x, y, z
        rot = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], "szyx")
        DT = np.eye(4)
        DT[:3, :3] = rot
        DT[:3, 3] = p
        return DT

    optim_counter = 0

    def pose_optimizer(x):
        nonlocal rotational_weight, T_end, optim_counter

        # print(x)
        # Render current pose
        DT = convert_to_pose(x)

        # OBJECT POSE
        current = np.dot(np.linalg.inv(T_end), DT)  # Transforms3DUtils.translation(x)
        current = np.linalg.inv(current)

        # CAMERA POSE
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
            # sift = cv2.ORB_create(nfeatures=2000)
            sift = cv2.SIFT_create(nfeatures=2000)
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
            max_distance = 0.0
            for m, n in matches:
                if m.distance > max_distance:
                    max_distance = m.distance
                if n.distance > max_distance:
                    max_distance = n.distance
                if m.distance < 0.5 * n.distance:
                    good.append([m])

            match_distances = 0.0
            for g in good:
                p1 = np.array(kp1[g[0].queryIdx].pt)
                p2 = np.array(kp2[g[0].trainIdx].pt)
                norm_distance = max_distance / g[0].distance

                match_distances += (np.linalg.norm(p1 - p2) * (1 / norm_distance)) ** 2

            match_distances /= len(good)

            n_match = 100
            match_distances += 100000 * max((n_match - len(good)) / n_match, 0)

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
            wr((stack * 255).astype(np.uint8))

            k = cv2.waitKey(1)
            if ord("q") == k:
                sys.exit(0)

            fitness = match_distances

            if optim_counter > warmap_epochs:
                fitness += 1000 * np.abs(start_image - end_image).mean()

            print("F", fitness, f"({optim_counter})", optim_counter > warmap_epochs)

            optim_counter += 1
            return fitness
        except:
            return np.inf
        # return (start_image - end_image).mean()

    rotational_weight = 1
    rotational_bound = rotational_weight * np.pi * 12
    translational_bound = 5

    # Choose optimizer -> https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
    # optimizer = ng.optimizers.CMA(
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
        budget=epochs,
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
