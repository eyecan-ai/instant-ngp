from re import U
import time
import typer as t
from pathlib import Path


def compute_from_instant(
    instant_folder: Path = t.Option(..., help="Folder with images and transform.json"),
    neural_twin_file: Path = t.Option(..., help="Neural Twin File"),
    spp: int = t.Option(1, help="Number of Samples per Pixel used for rendering"),
    ngp_build_folder: Path = t.Option("build", help="Folder in which Instant is built"),
):
    import sys

    sys.path.append(str(ngp_build_folder.absolute()))
    import pyngp as ngp  # type: ignore
    import msgpack
    import numpy as np
    import json

    # from pipelime.sequences.readers.base import ReaderTemplate
    from pipelime.sequences.readers.filesystem import UnderfolderReader
    from pipelime.sequences.writers.filesystem import UnderfolderWriterV2
    import cv2
    import rich
    from pathlib import Path

    # from pipelime.sequences.samples import SamplesSequence, PlainSample

    # Selecting mode
    mode = ngp.TestbedMode.Nerf

    # Getting the testbed for the selected mode
    testbed = ngp.Testbed(mode)

    # Loading desired training data
    # testbed.load_training_data(str(instant_folder))
    transforms = json.load(open(instant_folder / "transforms.json"))

    # Loading the network
    testbed.load_snapshot(str(neural_twin_file))

    # Loading the scene
    # ntwin: dict = msgpack.unpackb(neural_twin_file.read_bytes())
    # print(list(ntwin.keys()))

    # for k in ntwin.keys():
    #     if k != "snapshot":
    #         rich.print(k, ntwin[k])
    # print(list(ntwin["encoding"].keys()))
    # print(list(ntwin["snapshot"]["density_grid_size"].keys()))
    # # # Cropping the scene using the mask
    # testbed.render_aabb = ngp.BoundingBox(ntwin["bb_min"], ntwin["bb_max"])
    # testbed.render_aabb = ngp.BoundingBox([-1, -1, -1], [1, 1, 1])

    # # Setting intrinsics
    # testbed.set_camera_to_training_view(0)

    # # Setting the exposure
    testbed.exposure = 0

    # # Set the background to zero alpha
    # testbed.background_color = np.array([0, 0, 0, 0])

    # # Getting width and height
    downsample = 3
    width, height = transforms["w"] // downsample, transforms["h"] // downsample
    width, height = 224, 224
    print("SIZE", width, height)
    # dataset = UnderfolderReader(input_folder)

    ####################################################
    ## SLURIP ZOOP
    ####################################################
    # frame = transforms["frames"][0]
    # rich.print(np.matrix(frame["transform_matrix"])[:-1, :])

    # while True:
    #     t = time.perf_counter()

    #     angle = np.sin(t) * 30.0 + 90.0
    #     print(angle)

    #     testbed.fov_axis = 0
    #     testbed.fov = angle
    #     testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1, :])
    #     rendered_image = testbed.render(int(width), int(height), spp, True)

    #     rendered_image = rendered_image[:, :, :3]
    #     cv2.imshow("rendered", cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(1)

    ####################################################
    ## WASD
    ####################################################
    # frame = transforms["frames"][0]
    # T = np.array(frame["transform_matrix"])

    # dv = 0.01

    # while True:
    #     testbed.fov_axis = 0
    #     testbed.fov = 40

    #     testbed.set_nerf_camera_matrix(T[:-1, :])
    #     rendered_image = testbed.render(int(height), int(height), spp, True)

    #     rendered_image = rendered_image[:, :, :3]
    #     cv2.imshow("rendered", cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    #     k = cv2.waitKey(0)

    #     print(T)

    #     dt = np.eye(4)
    #     if k == ord("a"):
    #         dt[0, 3] = -dv
    #     elif k == ord("d"):
    #         dt[0, 3] = dv
    #     elif k == ord("w"):
    #         dt[1, 3] = -dv
    #     elif k == ord("s"):
    #         dt[1, 3] = dv

    #     T = T @ dt
    #     if k == ord("q"):
    #         break

    ####################################################
    ## Stereo matching
    ####################################################
    output_folder = Path("/tmp/render")
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    frame = transforms["frames"][0]
    T = np.array(frame["transform_matrix"])

    offset = np.eye(4)
    offset[1, 3] = 1.2

    T = T @ offset

    vt = 0.001
    dt = np.eye(4)
    dt[0, 3] = 0.25

    counter = 0
    cumulative_depth = None
    while True:
        testbed.fov_axis = 0
        testbed.fov = 40

        fx = width / np.tan(np.deg2rad(40 / 2))

        dt[0, 3] += vt

        def render_frame(pose):
            testbed.set_nerf_camera_matrix(pose[:-1, :])
            rendered_image = testbed.render(int(height), int(height), spp, True)
            rendered_image = rendered_image[:, :, :3]
            return rendered_image

        T_left = T
        T_right = T @ dt

        img_left = render_frame(T_left)
        reference_image = img_left.copy()
        img_right = render_frame(T_right)
        # img_left = (cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        # img_right = (cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        img_left = (img_left * 255).astype(np.uint8).copy()
        img_right = (img_right * 255).astype(np.uint8).copy()

        # disparity settings
        win_size = 5
        min_disp = -1
        max_disp = 63  # min_disp * 9
        num_disp = max_disp - min_disp  # Needs to be divisible by 16
        # Create Block matching object.
        stereo = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=64,
            blockSize=13,
        )  # 32*3*win_size**2)
        print("STEREO ok")

        # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        disparity = stereo.compute(img_left, img_right)

        depth = dt[0, 3] * fx / disparity
        max_depth = 1.0
        depth = np.clip(depth, 0, max_depth)

        if cumulative_depth is None:
            cumulative_depth = depth
        else:
            alpha = 0.9
            cumulative_depth = alpha * cumulative_depth + (1 - alpha) * depth

        print("DEPTH", depth.min(), depth.max())
        print("Disparity", dt[0, 3], "FX", fx)

        depth_color = (255 * cumulative_depth / max_depth).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_MAGMA)

        disparity = cv2.normalize(
            disparity,
            disparity,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

        print("OK", disparity.shape, disparity.min(), disparity.max())
        stack = np.hstack(
            (
                cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR),
                disparity,
            )
        )

        filename = output_folder / f"{str(counter).zfill(5)}.png"
        print(filename)
        counter += 1

        cv2.imwrite(str(filename), stack)

        cv2.imshow("stack", stack)
        cv2.imshow("depth", depth_color)

        k = cv2.waitKey(0)
        if k == ord("q"):
            break

        if k == ord("k"):
            import open3d as o3d

            print("COLOR", reference_image.shape)
            depth_image = o3d.geometry.Image(cumulative_depth.astype(np.float32))
            color_image = o3d.geometry.Image((reference_image * 255).astype(np.uint8))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image, convert_rgb_to_intensity=False
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    width, height, fx, fx, width // 2, height // 2
                ),
            )
            o3d.visualization.draw_geometries([pcd])
            print(pcd)


if __name__ == "__main__":
    t.run(compute_from_instant)


# ffmpeg images to video
# ffmpeg -r 30 -f image2 -s 1920x1080 -i /tmp/render/%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p /tmp/render/out.mp4
