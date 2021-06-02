import pickle
import typing
from typing import Any
from typeguard import typechecked
import os
import re
import cv2

# pickle utils
@typechecked
def save_object(obj: Any, filename: str) -> None:
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

@typechecked
def load_object(filename: str) -> Any:
    with open(filename, "rb") as input:  # note rb and not wb
        return pickle.load(input)

@typechecked
def sort_alphanumeric(data: list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

@typechecked
def write_video(
    image_dir: str, out_file: str, fps: int
):
    img_array = []
    im_list = os.listdir(image_dir)
    im_list = sort_alphanumeric(im_list)

    if ".DS_Store" in im_list:
        im_list.remove(".DS_Store")

    for filename in im_list:
        img = cv2.imread(os.path.join(image_dir, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        filename=out_file,
        fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps=fps,
        frameSize=size,
    )  # 15 fps

    cv2.VideoWriter()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()