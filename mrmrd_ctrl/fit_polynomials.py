"""This script should just take one data frame for a single session, 
and return a dataframe with curvature and polynomial coeffs.
What do we need: path for pandas data frame. read it. get chin. """
from datetime import datetime
import pandas as pd
import numpy as np
import os
from mrmrd_ctrl.utils.utils_IO import reshape_trial_dframe
from mrmrd_ctrl.utils.polyfitting import (
    fit_polynomials_per_frame,
    eval_polynomials_per_frame,
    compute_curvature_per_frame,
)
import typing
import datetime
import plotly

version = "V5"  # Note: V5 is updated for September 2021
path_to_sess_reconstruction_folder = (
    "/Volumes/sawtell-locker/C1/free/3d_reconstruction/{}".format(version)
)
path_to_video_folder = "/Volumes/sawtell-locker/C1/free/vids"  # videos and 2d tracking
sess_names = os.listdir(path_to_sess_reconstruction_folder)
interpolation_points = np.arange(5)  # since number of chin points = 5


def flag_well_trials(
    track_2d_dframe: pd.core.frame.DataFrame,
) -> pd.core.frame.DataFrame:
    """well is there if the summed confidence across 12 well keypoints is 5.0/12.0"""
    well_confidence = track_2d_dframe.filter(regex="well").filter(regex="confidence")
    assert len(well_confidence.columns) == 12
    return well_confidence.sum(axis="columns") > 5.0


def compute_inclusion_criteria(
    track_2d_dframe: pd.core.frame.DataFrame, dframe_3d: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """we include rows with high head confidence and no nans in 3d reconstruction"""
    low_head_confidence = track_2d_dframe["head_main_confidence"] < 0.5
    # Fede's method
    cols_to_take = dframe_3d.columns[
        dframe_3d.columns.get_level_values(level=0).isin(
            ["head", "chin_base", "chin_tip"]
        )
    ]
    is_nan_to_exclude = dframe_3d[cols_to_take].isnull().sum(axis="columns") > 0
    # a simpler approach that returns slightly different results
    # is_nan_to_exclude = (dframe_3d.isnull().sum(axis="columns") > 0).sum()
    rows_to_include = ~is_nan_to_exclude & ~low_head_confidence
    return rows_to_include


fish_names = [sess.split("_")[-1] for sess in sess_names]
datestrings = [sess.split("_")[0] for sess in sess_names]
datetime_dates = [
    datetime.date(int(ds[:4]), int(ds[4:6]), int(ds[6:])) for ds in datestrings
]
# # datetime_dates.sort()
# sorted_datestrings = []
# for d in datetime_dates:
#     print(d)
#     print(str(d).replace("-", ""))
#     sorted_datestrings.append(str(d).replace("-", ""))
# # TODO: need the session order. use datetime()?
# # TODO: final grand average is across fish too.

# creating the column indexes
unique_fish_names = list(np.unique(fish_names))
conds = ["well", "floor"]
iterables = [
    [
        "curvature",
        "second_order_coeff_x",
        "second_order_coeff_y",
        "second_order_coeff_z",
    ],
    ["median", "low", "high"],
]
mi_cols = pd.MultiIndex.from_product(iterables, names=["metric", "summary"])

# creating the row indexes
list_of_tuples = list(zip(fish_names, datetime_dates))  # names and dates
cond_list_long = conds * len(fish_names)  # ["floor", "well"] many times
cond_list_long.sort()  # sort to ["floor", ..., "floor", "well", ..., "well"]
list_with_conds = list(zip(list_of_tuples * 2, cond_list_long))
for ind, entry in enumerate(list_with_conds):
    list_with_conds[ind] = (
        entry[0][0],
        entry[0][1],
        entry[1],
    )  # the first two are tuples, and the last is a separate element
mi_index = pd.MultiIndex.from_tuples(
    list_with_conds, names=["fish", "date", "condition"]
)
dfmi = (
    pd.DataFrame(
        np.arange(len(mi_index) * len(mi_cols)).reshape((len(mi_index), len(mi_cols))),
        index=mi_index,
        columns=mi_cols,
        # dtype="Int64",  # for adding pd.NA if necessary
    ).sort_index()
    # .sort_index(axis=1) # date is sorted even without this line, see dfmi["curvature"].loc["Joao"]
)

# dfmi.loc["Sean"] # good
# dfmi.loc[("Sean", "20210116", "well")] # good
# dfmi.loc[("Sean", "20210116", "well"), ("curvature", "median")] # is good

sess_names = sess_names[ind:] # TODO: CAREFUL: hack just for now

for ind, sess in enumerate(sess_names):
    print(f"Analyzing Session {ind}/{len(sess_names)}")
    raw_date, fish_name = sess.split(
        "_"
    )  # that works for access: dfmi.loc[(fish_name, raw_date)]
    # date = datetime.date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:])) # not used for now
    # TODO 2: convert dates to integers for plotting, [1,2,3,4,5] or create an index or column saying session number
    # load 2D tracking to exclude points and determine well versus no well
    track_2d_dframe = pd.read_csv(
        os.path.join(path_to_video_folder, sess, "concatenated_tracking.csv"),
        header=[0],
    )
    # load 3D reconstruction which is the focus of this analysis
    dframe_3d = pd.read_csv(
        os.path.join(path_to_sess_reconstruction_folder, sess, "points_3d.csv"),
        header=[0, 1],
        index_col=0,
    )
    if len(dframe_3d) == len(track_2d_dframe) - 1:
        track_2d_dframe = track_2d_dframe.iloc[:-1, :]
    else:
        assert len(dframe_3d) == len(
            track_2d_dframe
        )  # frame nums should match, sometimes there's one frame difference, what should we exclude

    # flag well trials based on the 2d tracking
    is_well_condition = flag_well_trials(track_2d_dframe)
    # determine trials to exclude based on 2d and 3d criteria
    total_included_rows = compute_inclusion_criteria(track_2d_dframe, dframe_3d)

    print(
        "after exclusion, analyzing {}/{} frames!".format(
            total_included_rows.sum(), len(total_included_rows)
        )
    )

    # removing bad trials
    cleaned_dframe_3d = dframe_3d.loc[total_included_rows, :]
    cleaned_is_well_condition = is_well_condition.loc[total_included_rows]
    assert (
        cleaned_dframe_3d.shape[0] <= dframe_3d.shape[0]
        and cleaned_dframe_3d.shape[1] == dframe_3d.shape[1]
    )
    # start with the curvature analyses (no exclusion yet?)
    chin_trial_arr, bp_names = reshape_trial_dframe(cleaned_dframe_3d)
    assert bp_names == ["chin_base", "chin1_4", "chin_half", "chin3_4", "chin_tip"]
    # first, calculate polynomial coefficients per frame
    coeffs_all_frames = fit_polynomials_per_frame(data_arr=chin_trial_arr, degree=2)
    # second, evaluate the polynomials across the five points of the chin
    evals = eval_polynomials_per_frame(coeffs_all_frames, interpolation_points)
    # compute curvatures
    curvatures = compute_curvature_per_frame(
        coeffs_all_frames=coeffs_all_frames, interpolation_points=interpolation_points
    )
    # separate curvature to well/floor conditions
    well_curvature = curvatures[cleaned_is_well_condition, :]
    floor_curvature = curvatures[~cleaned_is_well_condition, :]
    curvature_dict = {"well": well_curvature, "floor": floor_curvature}

    # separate coeffs to the above conditions
    well_coeffs = coeffs_all_frames[cleaned_is_well_condition, :, :]
    floor_coeffs = coeffs_all_frames[~cleaned_is_well_condition, :, :]

    coeff_dict = {"well": well_coeffs, "floor": floor_coeffs}
    dim_list = ["x", "y", "z"]
    degree_list = [2, 1, 0]
    # basically all these values turn out to be zeros. I checked that
    # TODO: make sure you iterate properly over the coeffs, and save them in the proper well/floor cond
    for well_cond, coeff_arr in coeff_dict.items():
        for i in range(coeff_arr.shape[1]):  # (x,y,z) dim
            if coeff_arr.shape[0] < 1000:  # less than 1000 frames
                median, low, high = pd.NA, pd.NA, pd.NA
            else:
                second_order_coeffs = coeff_arr[
                    :, i, 0
                ]  # 0th index is highest degree. in this case, second.
                median = np.nanmedian(second_order_coeffs)
                low, high = np.nanpercentile(second_order_coeffs, [25.0, 75.0])

            dfmi.loc[
                (fish_name, raw_date, well_cond),
                ("second_order_coeff_{}".format(dim_list[i]), "median"),
            ] = median
            dfmi.loc[
                (fish_name, raw_date, well_cond),
                ("second_order_coeff_{}".format(dim_list[i]), "low"),
            ] = low
            dfmi.loc[
                (fish_name, raw_date, well_cond),
                ("second_order_coeff_{}".format(dim_list[i]), "high"),
            ] = high
    # take grand average of curvatures, with percentiles
    for well_cond, curvatures in curvature_dict.items():
        if curvatures.shape[0] < 1000:
            median, low, high = pd.NA, pd.NA, pd.NA
        else:
            median = np.nanmedian(curvatures.flatten())
            assert median is not None and median > 0
            low, high = np.nanpercentile(curvatures.flatten(), [25.0, 75.0])
            assert low < high
        dfmi.loc[(fish_name, raw_date, well_cond), ("curvature", "median")] = median
        dfmi.loc[(fish_name, raw_date, well_cond), ("curvature", "low")] = low
        dfmi.loc[(fish_name, raw_date, well_cond), ("curvature", "high")] = high

# dfmi.loc[(fish_name, raw_date)]
if not os.path.exists("summaries"):
    os.mkdir("summaries")
dfmi.to_csv("summaries/curvature_dframe.csv", index=True)

# later we'll compute the curvature across fish and conds!
# dfmi_loaded = pd.read_csv("summaries/curvature_dframe.csv", header= [0, 1], index_col=[0,1,2])
# dfmi_loaded.loc[(fish_name, str(datetime.date(2021, 1, 24)))] that works
