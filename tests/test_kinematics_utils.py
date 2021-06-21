import pandas as pd
import numpy as np
from mrmrd_ctrl.utils.kinematics import compute_speed, compute_accel
import pytest

@pytest.fixture
def generate_fake_dframe():
		df = pd.DataFrame(np.random.randint(0,100,size=(15, 4)), columns=list('ABCD'))
		return df

def test_compute_speed(generate_fake_dframe):
	df = generate_fake_dframe
	out_dframe = compute_speed(df, 300)
	assert out_dframe.shape == df.shape
	assert (out_dframe.iloc[0, :] != out_dframe.iloc[0, :]).all()

def test_compute_accel(generate_fake_dframe):
	df = generate_fake_dframe
	out_dframe = compute_accel(df, 300)
	assert out_dframe.shape == df.shape
	assert (out_dframe.iloc[:2, :].values.flatten() != out_dframe.iloc[:2, :].values.flatten()).all()

