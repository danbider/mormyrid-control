import numpy as np
import pytest
from mrmrd_ctrl.mv_regression import infer_beta
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple

def _add_diag_noise(X):
	diag_inds = (np.arange(X.shape[0]), np.arange(X.shape[0]))
	X[diag_inds] += 0.1
	return X

@pytest.fixture
def set_const_regression_params():
	T = 200
	K = 5
	L = 3
	return (T,K,L)

def test_params(set_const_regression_params: Tuple) -> None:
	assert(set_const_regression_params == (200,5,3))

@pytest.fixture
def make_fake_regression_dataset(set_const_regression_params):
	# make up data for noiseless linear regression
	(T,K,L) = set_const_regression_params
	reg_dict = {}
	reg_dict["beta_true"] = np.array([[0.1, 0.2, 0.3, 2.4, 11.1], [4.1, 5.2, 6.3, 12.1, 9.9], [2.2, 3.0, 5.0, 23.0, 10.0]])
	assert(reg_dict["beta_true"].shape == (L, K))
	X = np.float64(np.tile(np.arange(T), (K, 1)))
	X = _add_diag_noise(X)
	Y = np.dot(reg_dict["beta_true"], X)
	reg_dict["X"] = X
	reg_dict["Y"] = Y
	return reg_dict
#
def test_mv_regression(make_fake_regression_dataset: Dict):
	# compare inferred to true beta
	reg_dict = make_fake_regression_dataset
	beta_inferred = infer_beta(reg_dict["X"], reg_dict["Y"])
	assert((np.round(np.abs(beta_inferred - reg_dict["beta_true"]),5) == 0).all())
	reg = LinearRegression(fit_intercept=False).fit(reg_dict["X"].T, reg_dict["Y"].T)
	assert((np.round(np.abs(reg.coef_  - reg_dict["beta_true"]),5) == 0).all()) # make sure that sklearn is doing ok before comparing to it
	assert((np.round(np.abs(reg.coef_  - beta_inferred),5) == 0).all()) # compare infer_beta() to sklearn
