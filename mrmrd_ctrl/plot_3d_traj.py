import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from plotly.subplots import make_subplots


def slice_dframe(dframe, start_ind, num_frames):
    end_ind = start_ind + num_frames
    assert (len(dframe) > start_ind) & (
        len(dframe) > end_ind
    )  # assert that we're within bounds.
    return dframe.iloc[start_ind:end_ind, :]


version = "V5"  # Note: V5 is updated for September 2021
path_to_sess_reconstruction_folder = (
    "/Volumes/sawtell-locker/C1/free/3d_reconstruction/{}".format(version)
)
path_to_video_folder = "/Volumes/sawtell-locker/C1/free/vids"  # videos and 2d tracking
# sess_names = os.listdir(path_to_sess_reconstruction_folder)

# number of frames to use after start_ind
num_frames = {"floor": 2000, "well": 2000}

# pre-selected frames from four videos
start_ind = {}
start_ind["floor"] = {"pre": 45038, "post": 1027018}
start_ind["well"] = {"pre": 4673, "post": 200875}

video_names = {}
video_names["floor"] = {"pre": "20201104_Joao", "post": "20201114_Joao"}
video_names["well"] = {"pre": "20201106_Joao", "post": "20201120_Joao"}

camera = dict(
    up=dict(
        x=0, y=-1, z=0
    ),  # the up-down axis is defined by y coordinate. bigger values are downwards
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.0, y=-0.5, z=-1.5),  # peaking as an experimenter into the tank
)
# # # # test camera
# data = go.Scatter3d(
#                 x=df_chopped["chin_tip"]["x"],
#                 y=df_chopped["chin_tip"]["y"],
#                 z=df_chopped["chin_tip"]["z"],
#                 marker=dict(
#                     size=2,
#                     color=colors,
#                     colorscale="blackbody",
#                     colorbar=dict(thickness=20, title="Time (s)"),
#                 ))

# fig = go.Figure(data=data)
# fig.update_layout(scene_camera = camera)
# fig.show()

traces = []
rows = [1, 1, 2, 2]
columns = [1, 2, 1, 2]
fps = 300  # true parameter from experiment

fig = make_subplots(
    rows=2,
    cols=2,
    row_titles=("floor", "well"),
    column_titles=("pre-lesion", "post-lesion"),
    specs=[
        [{"type": "scene"}, {"type": "scene"}],
        [{"type": "scene"}, {"type": "scene"}],
    ],
    horizontal_spacing=1.0,
    vertical_spacing=1.0,
)
counter = 0
# TODO: multiple colorbars: see https://community.plotly.com/t/subplots-of-two-heatmaps-overlapping-text-colourbar/38587/5
for cond, dicts in video_names.items():
    colors = np.arange(num_frames[cond]) * (
        1.0 / fps
    )  # these should be identical across conds because we plot just one colorbar now
    for lesion_cond, video_name in dicts.items():
        df = pd.read_csv(
            os.path.join(
                path_to_sess_reconstruction_folder, video_name, "points_3d.csv"
            ),
            header=[0, 1],
        )
        df_chopped = slice_dframe(df, start_ind[cond][lesion_cond], num_frames[cond])
        traces.append(
            go.Scatter3d(
                x=df_chopped["chin_tip"]["x"],
                y=df_chopped["chin_tip"]["y"],
                z=df_chopped["chin_tip"]["z"],
                marker=dict(
                    size=2,
                    color=colors,
                    colorscale="blackbody",
                    colorbar=dict(thickness=20, title="Time (s)"),
                ),
                showlegend=False,
            )
        )

        fig.add_trace(traces[-1], row=rows[counter], col=columns[counter])
        counter += 1
        print(counter)
fig.update_scenes(camera=camera)  # update all viewing angles
fig.show()  # it'll open on browser

fig.update_layout(
    autosize=False, width=1200, height=1000, font=dict(size=14)
)  # this controls the xyz axes
fig.update_annotations(
    font_size=40
)  # these are the row and column titles, see https://community.plotly.com/t/setting-subplot-title-font-sizes/46612


if not os.path.exists("images"):
    os.mkdir("figs")
fig.write_image("figs/3d_fig.png")
