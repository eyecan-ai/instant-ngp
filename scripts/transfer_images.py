from pathlib import Path
import cv2
import numpy as np
import typer


def transfer(image: np.ndarray, mode: str = "canny") -> np.ndarray:

    if mode == "canny":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(image, 100, 200)
        h, w = image.shape[:2]
        output = np.zeros((h, w, 4), dtype=np.uint8)
        output[:, :, 0] = canny
        output[:, :, 1] = canny
        output[:, :, 2] = canny
        output[:, :, 3] = canny
        return output

    else:
        raise NotImplementedError("Mode not implemented")


def transfer_images(
    folder: Path = typer.Option(..., help="Images folder"),
    output_folder: Path = typer.Option(..., help="Output folder"),
    extension: str = typer.Option("jpg", help="Images extension"),
    output_extension: str = typer.Option("png", help="Output extension"),
    mode: str = typer.Option("canny", help="Transfer Mode"),
):

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    filenames = folder.glob("*.{}".format(extension))

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    for fn in filenames:
        image = cv2.imread(str(fn))

        transferred = transfer(image, mode)
        cv2.imshow("image", transferred)
        cv2.waitKey(0)

        output_filename = output_folder / fn.name.replace(extension, output_extension)
        print(output_filename)
        cv2.imwrite(str(output_filename), transferred)


if __name__ == "__main__":
    typer.run(transfer_images)
