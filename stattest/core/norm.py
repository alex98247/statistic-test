from scipy.stats import norm


def generate(size, mean=0, var=1):
    return norm.rvs(size=size, loc=mean, scale=var)
