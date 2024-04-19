from scipy.stats import expon


def generate_expon(size, l=1):
    scale = 1 / l
    return expon.rvs(size=size, scale=scale)


def cdf_expon(rvs, l=1):
    scale = 1 / l
    return expon.cdf(rvs, scale=scale)
