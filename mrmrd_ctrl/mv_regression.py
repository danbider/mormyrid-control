from typing import Any, Dict, List, Optional
from typeguard import typechecked
import numpy as np

@typechecked
def infer_beta(X: np.array, Y: np.array) -> np.array:
    return np.linalg.multi_dot([Y, X.transpose(), np.linalg.inv(np.dot(X, X.transpose()))])
