import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import typing
from typing import Any
from typeguard import typechecked
import pickle
from utils.utils_IO import save_object, load_object
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--general_3d_recon_path", type=str, default="/Volumes/sawtell-locker/C1/free/3d_reconstruction/V2", help="path to 3d recon files")
parser.add_argument("--general_video_path", type=str, default="/Volumes/sawtell-locker/C1/free/vids", help="path to the folders with vids and net preds")
parser.add_argument("--csv_trial_path", type=str, default="/Volumes/sawtell-locker/C1/free/C1_prey_capture_events.csv", help="path to file indicating which frames to analyze")
args, _ = parser.parse_known_args()

# assert correct paths:
assert(os.path.isdir(args.general_3d_recon_path))
assert(os.path.isdir(args.general_video_path))
assert(os.path.isfile(args.csv_trial_path))

@typechecked
def get_vid_folder_name(general_video_path: str, date: str, name: str) -> str:
    '''return a path for videos folder from names '''
    full_path = os.path.join(general_video_path , date + '_' + name)
    assert(os.path.isdir(full_path))
    return full_path

# @typechecked
# def make_empty_list_of_lists(list_length: int) -> list[list]:
#     list_of_lists = list(range(list_length))
#     for i in range(len(list_of_lists)):
#         list_of_lists[i] = []
#     return list_of_lists

@typechecked
def make_empty_hierarchical_dict(unique_names: list, unique_conds: list)-> dict:
    trial_dict = {}
    for name in unique_names:
        trial_dict[name] = {}
        for cond in unique_conds:
            trial_dict[name][cond] = []
    return trial_dict

@typechecked
def concat_video_paths(dframe: pd.core.frame.DataFrame, general_video_path: str) -> list[str]:
    list_concat = []
    for i in range(dframe.shape[0]): # loop over rows and merge date and fish strings
        list_concat.append(get_vid_folder_name(general_video_path,
                                str(dframe["Date"][i]),
                                    dframe["Fish"][i]))
    return list_concat

if __name__ == "__main__":
    # read the csv with trial numbers
    df_w_frame_nums = pd.read_csv(args.csv_trial_path, header=0)

    unique_names = pd.unique(df_w_frame_nums["Fish"])
    unique_conds = pd.unique(df_w_frame_nums["Condition"])

    # fill inds dict with the frames to keep
    inds_dict = make_empty_hierarchical_dict(list(unique_names), list(unique_conds))
    frame_pad = 0
    for j in range(df_w_frame_nums.shape[0]):
        inds_dict[df_w_frame_nums["Fish"][j]][df_w_frame_nums["Condition"][j]].append(
            np.arange(df_w_frame_nums["Original Vid Frame START"][j] - frame_pad,
                      df_w_frame_nums["Original Vid Frame STOP"][j] + 1 + frame_pad))

    # grab the relevant 3d path and video path
    points_3d_path_dict = make_empty_hierarchical_dict(list(unique_names), list(unique_conds))
    video_path_dict = make_empty_hierarchical_dict(list(unique_names), list(unique_conds))

    for i, name in enumerate(unique_names):
        for j, cond in enumerate(unique_conds):
            inds = (df_w_frame_nums["Condition"] == cond) & (df_w_frame_nums["Fish"] == name)
            curr_date = pd.unique(df_w_frame_nums.loc[inds]["Date"])
            assert (len(curr_date) == 1) # for now assuming one session per fish X condition
            points_3d_path_dict[name][cond] = os.path.join(args.general_3d_recon_path, str(curr_date[0]) + '_' + name,
                                                   'points_3d.csv')
            assert(os.path.isfile(points_3d_path_dict[name][cond]))
            video_path_dict[name][cond] = os.path.join(args.general_video_path, str(curr_date[0]) + '_' + name)
            assert(os.path.isdir(video_path_dict[name][cond])) # directory with videos (w or w/o labels)

    # make the actual dict with trials
    trial_dict = make_empty_hierarchical_dict(list(unique_names), list(unique_conds))
    for i, name in enumerate(unique_names):
        for j, cond in enumerate(unique_conds):
            # read file
            print("Fish: %s, condition: %s" % (name, cond))
            print("analyzing file %s ..." % points_3d_path_dict[name][cond])
            csv_3d = pd.read_csv(points_3d_path_dict[name][cond], header=[0, 1])
            # take the relevant slices and save them as separate trials
            for k, ind_arr in enumerate(inds_dict[name][cond]):
                trial_dict[name][cond].append(csv_3d.iloc[ind_arr, :])

    # make dict with all relevant dicts for further analysis
    data_dict = {}
    data_dict["trials"] = trial_dict
    data_dict["inds"] = inds_dict
    data_dict["points_3d_paths"] = points_3d_path_dict
    data_dict["video_paths"] = video_path_dict

    dict_save_name = os.path.join(args.general_3d_recon_path, "data_dict")
    save_object(data_dict, dict_save_name)
    print("saved a dictionary at: \n %s" % dict_save_name)
