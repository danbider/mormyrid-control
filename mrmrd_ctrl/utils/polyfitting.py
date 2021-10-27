from typeguard import typechecked
import numpy as np
from tqdm import tqdm
from typing import Callable
from scipy.signal import medfilt
import matplotlib


# TODO: test
def compute_curvature_function(x, y, z):
    # r = (x,y,z) each coeffs of a quadratic spline
    # numerator = 2 * np.asarray([y[1]*z[2] - y[2]*z[1], - x[1]*z[2] + x[2]*z[1], x[1]*y[2] - x[2]*y[1]])
    numerator = (
        (2 * (y[1] * z[0] - y[0] * z[1])) ** 2
        + (2 * (-x[1] * z[0] + x[0] * z[1])) ** 2
        + (2 * (x[1] * y[0] - x[0] * y[1])) ** 2
    )
    numerator = numerator ** (1.0 / 2.0)
    # numerator = np.linalg.norm(numerator)
    def curve_func(t):
        # return numerator / (np.linalg.norm(np.asarray([x[1] + 2*x[2] * t, y[1] + 2*y[2] * t, z[1] + 2*z[2] * t])) ** 3)
        denominator = (
            (x[1] + 2 * x[0] * t) ** 2
            + (y[1] + 2 * y[0] * t) ** 2
            + (z[1] + 2 * z[0] * t) ** 2
        ) ** (3.0 / 2.0)
        return numerator / denominator

    return curve_func


@typechecked
def fit_3D_polynomial(data: np.ndarray, degree: int = 2) -> np.ndarray:
    assert data.shape[-1] == 3  # (x,y,z) coords
    t = np.arange(
        data.shape[0]
    )  # we use this t to trace (x(t), y(t), z(t)). TODO: could have one fewer DOF
    poly_coeffs = np.zeros((3, degree + 1))
    for i in range(data.shape[-1]):
        poly_coeffs[i, :] = np.polyfit(t, data[:, i], degree)
    return poly_coeffs


# fit polynomial to a sequence of time pts
@typechecked
def fit_polynomials_per_frame(data_arr: np.ndarray, degree: int = 2) -> np.ndarray:
    assert data_arr.shape[-1] == 3
    coeffs_all_frames = np.zeros((data_arr.shape[0], 3, degree + 1))
    for frame_idx in tqdm(range(data_arr.shape[0])):
        coeffs_all_frames[frame_idx, :, :] = fit_3D_polynomial(
            data=data_arr[frame_idx, :, :], degree=2
        )
    return coeffs_all_frames


# eval polynomials
@typechecked
def eval_3D_polynomials(
    poly_coeffs: np.ndarray, interpolation_points: np.ndarray
) -> np.ndarray:
    assert poly_coeffs.shape[0] == 3  # (x,y,z) dimensions
    evals = np.zeros((3, len(interpolation_points)))
    for dim in range(poly_coeffs.shape[-1]):
        evals[dim, :] = np.polyval(poly_coeffs[dim, :], interpolation_points)
    return evals


# eval at a sequence of time points
@typechecked
def eval_polynomials_per_frame(
    coeffs_all_frames: np.ndarray, interpolation_points: np.ndarray
) -> np.ndarray:
    assert coeffs_all_frames.shape[1] == 3  # 3 (x,y,z) dims
    evals_all_frames = np.zeros(
        (
            coeffs_all_frames.shape[0],
            coeffs_all_frames.shape[1],
            len(interpolation_points),
        )
    )
    for frame_idx in tqdm(range(coeffs_all_frames.shape[0])):
        evals_all_frames[frame_idx, :, :] = eval_3D_polynomials(
            poly_coeffs=coeffs_all_frames[frame_idx, :, :],
            interpolation_points=interpolation_points,
        )
    return evals_all_frames


@typechecked
def compute_curvature_per_frame(
    compute_curvature_function: Callable = compute_curvature_function,
    coeffs_all_frames: np.ndarray = None,
    interpolation_points: np.ndarray = None,
) -> np.ndarray:
    curvatures = np.zeros((coeffs_all_frames.shape[0], len(interpolation_points)))
    for frame_idx in tqdm(range(coeffs_all_frames.shape[0])):
        curvature_func = compute_curvature_function(
            coeffs_all_frames[frame_idx, 0, :],
            coeffs_all_frames[frame_idx, 1, :],
            coeffs_all_frames[frame_idx, 2, :],
        )
        curvatures[frame_idx, :] = curvature_func(interpolation_points)
    # NOTE: the first value is high; should be discarded
    return curvatures


@typechecked
def filter_coordinates(
    raw_numpy_arr: np.ndarray, coordinate_dims: int = 3, medfilt_kernel_size: int = 11
) -> np.ndarray:
    filtered_arr = np.zeros_like(raw_numpy_arr)
    assert (
        raw_numpy_arr.shape[1] % coordinate_dims == 0
    )  # make sure the number of cols is divisible by coordinate dims
    num_bodyparts = raw_numpy_arr.shape[1] // coordinate_dims
    for i in range(num_bodyparts):
        col_inds = np.arange(
            i + i * (coordinate_dims - 1),
            i + i * (coordinate_dims - 1) + coordinate_dims,
        )
        filtered_arr[:, col_inds] = medfilt(
            raw_numpy_arr[:, col_inds], medfilt_kernel_size
        )
    return filtered_arr


@typechecked
def summarize_curvature(
    curvature_arr: np.ndarray, metric: str = "mean", differences: bool = False
) -> np.ndarray:
    if differences:  # work with the frame-by-frame differences to account for movement
        num_rows_pre_diff = curvature_arr.shape[0]
        curvature_arr = np.diff(curvature_arr, axis=0)
        assert curvature_arr.shape[0] == num_rows_pre_diff - 1
    if metric == "mean":
        return np.nanmean(curvature_arr, axis=0)
    if metric == "median":
        return np.nanmedian(curvature_arr, axis=0)
    if metric == "max":
        return np.nanmax(curvature_arr, axis=0)
    if metric == "stderr":
        return np.nanstd(curvature_arr, axis=0) / np.sqrt(curvature_arr.shape[0])
    else:
        raise (ValueError('The requested summary "%s" is not supported.' % metric))
