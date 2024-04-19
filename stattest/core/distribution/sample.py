from scipy.stats import moment as scipy_moment
import numpy as np


def moment(a, moment=1, center=None):
    scipy_moment(a=a, moment=moment, center=center)


def central_moment(a, moment=1):
    return scipy_moment(a=a, moment=moment, center=np.mean(a, axis=0))
