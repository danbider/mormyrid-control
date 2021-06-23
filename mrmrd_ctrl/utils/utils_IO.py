import pickle
import typing
from typing import Any
from typeguard import typechecked
import os
import re
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
def write_video_from_directory(image_dir: str, out_file: str, fps: int):
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


@typechecked
def write_video_from_list(img_array: list, out_file: str, fps: int):

    height, width, layers = img_array[0].shape
    size = (width, height)

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


# TODO: test
def read_frame_at_idx(video_path, frame_idx):
    vidcap = cv2.VideoCapture(str(video_path))
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count >= frame_idx:
            return image
        count += 1

    if count < frame_idx:
        print(f"Request frame {frame_idx} is too large")
        return None


def plot_chin_spline(chin_trial_arr, poly_evals, bp_names, out_file, start_frame=0, end_frame=-1):
    img_array = []

    # Get global mins/maxes for plotting
    x_min = np.min(chin_trial_arr[:, :, 0])
    y_min = np.min(chin_trial_arr[:, :, 1])
    z_min = np.min(chin_trial_arr[:, :, 2])

    x_max = np.max(chin_trial_arr[:, :, 0])
    y_max = np.max(chin_trial_arr[:, :, 1])
    z_max = np.max(chin_trial_arr[:, :, 2])
    
    if end_frame == -1:
        end_frame = chin_trial_arr.shape[0]

    for frame_idx in tqdm(range(start_frame, end_frame)):
        chin_frame_points = chin_trial_arr[frame_idx]

        x = chin_frame_points[:, 0]
        x -= x[0]
        y = chin_frame_points[:, 1]
        y -= y[0]
        z = chin_frame_points[:, 2]
        z -= z[0]

        xi = poly_evals[frame_idx, 0, :]
        xi -= xi[0]
        yi = poly_evals[frame_idx, 1, :]
        yi -= yi[0]
        zi = poly_evals[frame_idx, 2, :]
        zi -= zi[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs=xi, ys=yi, zs=zi, c="b")
        ax.scatter(x, y, z, c="r")

        # ax.scatter(x_curve, y_curve, z_curve, c="r")
        ax.view_init(elev=-90, azim=-90)
        # ax.title.set_text(f"$\kappa$: {max_curve}")

        for i, txt in enumerate(range(chin_frame_points.shape[0])):
            if "base" in bp_names[i] or "end" in bp_names[i]:
                ax.text(x[i], y[i], z[i], bp_names[i])

        # ax.axes.set_xlim3d([-1, 1])
        # ax.axes.set_ylim3d([-1, 1])
        # ax.axes.set_zlim3d([-1, 1])

        ax.axes.set_xlim3d([x_min, x_max])
        ax.axes.set_ylim3d([y_min, y_max])
        ax.axes.set_zlim3d([z_min, z_max])

        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")
        ax.xaxis.set_tick_params(labelsize=0)
        ax.yaxis.set_tick_params(labelsize=0)
        ax.zaxis.set_tick_params(labelsize=0)
        ax.set_axis_off()

        fig.canvas.draw()
        chin_plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chin_plot_image = chin_plot_image.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        # Convert from RGB to BGR
        chin_plot_image = chin_plot_image[:, :, ::-1]
        plt.close("all")
        img_array.append(chin_plot_image)

        #! DON'T DELETE: To be used for plotting alongside image frames
        """
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        chin_plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chin_plot_image = chin_plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        video_frame = read_frame_at_idx(video_path, frame_idx)
        # reproj_frame = read_frame_at_idx(reproj_path, frame_idx)
        width = chin_plot_image.shape[1]
        height = chin_plot_image.shape[0]
        dim = (width, height)
        #vis = np.concatenate((chin_plot_image, cv2.resize(video_frame, dim), cv2.resize(reproj_frame, dim)), axis=1)
        vis = np.concatenate((chin_plot_image, cv2.resize(video_frame, dim)), axis=1)
        cv2.imwrite(str(save_dir / f"frame_{frame_idx}.png"), vis)
        """

    write_video_from_list(img_array=img_array, out_file=out_file, fps=50)
