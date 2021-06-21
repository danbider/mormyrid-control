import pandas as pd
import typing
from typeguard import typechecked

@typechecked
def compute_speed(dframe: pd.core.frame.DataFrame, fps: int) -> pd.core.frame.DataFrame:
		'''compute difference and divide by fps. units: a.u / s'''
		return dframe.diff().abs().div(fps)

@typechecked
def compute_accel(dframe: pd.core.frame.DataFrame, fps: int) -> pd.core.frame.DataFrame:
		'''compute difference and divide by fps. units: a.u / s'''
		speed = compute_speed(dframe, fps)
		return compute_speed(speed, fps)
