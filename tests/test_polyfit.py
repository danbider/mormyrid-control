import numpy as np
import pytest
from mrmrd_ctrl.utils.polyfitting import fit_3D_polynomial, filter_coordinates
from mrmrd_ctrl.utils.utils_IO import compute_hdi, load_object, reshape_trial_dframe


@pytest.fixture
def fake_poly_data():
    x = np.arange(5) + np.random.normal(size=(5,), scale=0.1)
    y = 6.0 + 1 * x + 5 * x ** 2
    z = 3.0 + 4 * x + 2 * x ** 2
    return (x, y, z)


# TODO: finalize test from the chin_spline notebook
def test_polyfit(fake_poly_data) -> None:
    x, y, z = fake_poly_data
    poly_coef = np.polyfit(x, z, 2)  # coefs come in a different order
    assert (np.abs(poly_coef - np.array([2.0, 4.0, 3.0])) < 0.000001).all()


def test_our_polyfit(fake_poly_data) -> None:
    x, y, z = fake_poly_data
    test_arr = np.stack([x, y, z]).T
    t = np.arange(test_arr.shape[0])
    coeffs_out = fit_3D_polynomial(data=test_arr, degree=2)
    poly_coef_manual = np.polyfit(
        t, y, 2
    )  # just look at the y coordinate, and make sure it returns the same results
    assert ((np.abs(poly_coef_manual - coeffs_out[1, :])) < 0.000001).all()


def test_filtering_reshaping() -> None:
    raw_numpy_arr = np.random.normal(size=(1000, 15))
    filtered_test_arr = np.zeros_like(raw_numpy_arr)
    coordinate_dims = (
        3  # 3d coords. cols of raw_numpy_arr should be divisible by that number
    )
    num_bodyparts = raw_numpy_arr.shape[1] // coordinate_dims
    assert type(raw_numpy_arr.shape[1] // coordinate_dims) is int
    for i in range(num_bodyparts):
        col_inds = np.arange(
            i + i * (coordinate_dims - 1),
            i + i * (coordinate_dims - 1) + coordinate_dims,
        )
        filtered_test_arr[:, col_inds] = raw_numpy_arr[:, col_inds]
    assert (abs(filtered_test_arr[:] - raw_numpy_arr[:]) < 0.001).all()


def test_filtering_equality() -> None:
    # assert that
    raw_numpy_arr = np.random.normal(size=(1000, 15))
    filtered_arr = filter_coordinates(
        raw_numpy_arr, coordinate_dims=1, medfilt_kernel_size=1
    )
    assert (abs(filtered_arr[:] - raw_numpy_arr[:]) < 0.00001).all()


def test_compute_hdi() -> None:
    nums = np.tile(np.random.permutation(np.arange(0, 101)), (3, 1)).T
    assert (compute_hdi(nums)[0] == np.array([2.5, 2.5, 2.5])).all()
    assert (compute_hdi(nums)[1] == np.array([97.5, 97.5, 97.5])).all()


def test_reshape_trial_dframe():
    trial_number = 0
    data_dict = load_object(
        "/Volumes/sawtell-locker/C1/free/3d_reconstruction/V2/data_dict"
    )  # local path to volume from Dan's machine
    example_trial = data_dict["trials"]["Joao"]["pre"][trial_number]
    chin_trial_arr, bp_names = reshape_trial_dframe(example_trial)
    assert chin_trial_arr.shape[1] == 5 and chin_trial_arr.shape[2] == 3
    assert bp_names == ["chin_base", "chin_1_4", "chin_mid", "chin_3_4", "chin_end"]
