from pathlib import Path
import numpy as np
import rich
import cv2
import typer as t


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


def combine_multi_video(
    multi_video_folder: str = t.Option(
        ...,
        help="Input folder with raw frames subfolders",
    ),
    output_folder: str = t.Option(
        ...,
        help="Output folder for processed frames",
    ),
):

    frame_writer = FrameWriter(output_folder)
    folder = Path(multi_video_folder)
    subfolders = list(sorted([x for x in folder.iterdir() if x.is_dir()]))

    carousel_tags = ["0001", "0010", "0100", "1000", "1111"]

    filesmap = {}
    total_size = np.inf
    tags = set()
    for subfolder in subfolders:
        files = list(sorted([x for x in subfolder.iterdir() if x.is_file()]))
        filesmap[subfolder.name] = files
        tags.add(subfolder.name)
        if len(files) < total_size:
            total_size = len(files)

    # SET The carousel in a specific frame id
    def precise_time_carousel(
        timeframes_map: dict,
        start_id: int,
        duration: int = 10,
        repeats: int = 2,
        tags: list = carousel_tags,
    ):

        for idx in range(len(tags) * repeats):
            current_idx = start_id + idx * duration
            timeframes_map[current_idx] = tags[idx % len(tags)]

    # SET a carousel along the whole video
    def continuous_carousel(
        timeframes_map: dict,
        size: int,
        start_id: int = 0,
        step: int = 100,
        tags: list = carousel_tags,
    ):

        counter = 0
        for idx in range(start_id, size, step):
            timeframes_map[idx] = tags[counter % len(tags)]
            counter += 1

    ############################
    # Carousel on waypoints
    ############################
    # timeframes_map = {0: "1111"}
    # duration = 4
    # # The following start id must be tuned to the waypoints watching/analyzing manually
    # # the recorded raw frames. E.G. 100 is the first frame where viewpoints stop changing
    # # so do the 260 and so on.
    # precise_time_carousel(timeframes_map, 100, duration=duration)
    # precise_time_carousel(timeframes_map, 260, duration=duration)
    # precise_time_carousel(timeframes_map, 400, duration=duration)
    # precise_time_carousel(timeframes_map, 560, duration=duration)
    # precise_time_carousel(timeframes_map, 715, duration=duration)
    # precise_time_carousel(timeframes_map, 865, duration=duration)
    # precise_time_carousel(timeframes_map, 1000, duration=duration)
    # precise_time_carousel(timeframes_map, 1165, duration=duration)
    # precise_time_carousel(timeframes_map, 1300, duration=duration)

    ############################
    # Continuous carousel
    ############################
    timeframes_map = {0: "1111"}
    # continuous_carousel(timeframes_map, size=total_size, step=5)

    # Process frames
    for i in range(total_size):
        if i in timeframes_map:
            tag = timeframes_map[i]

        image = cv2.imread(str(filesmap[tag][i]))
        frame_writer(image)
        cv2.imshow("image", image)
        cv2.waitKey(10)


if __name__ == "__main__":
    t.run(combine_multi_video)
