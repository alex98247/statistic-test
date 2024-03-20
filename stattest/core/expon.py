from scipy.stats import expon


def generate(size, l=1):
    scale = 1 / l
    return expon.rvs(size=size, scale=scale)
