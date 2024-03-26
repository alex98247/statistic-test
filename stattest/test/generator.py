from stattest.core.beta import generate_beta
from stattest.core.cauchy import generate_cauchy
from stattest.core.laplace import generate_laplace
from stattest.core.logistic import generate_logistic
from stattest.core.student import generate_t
from stattest.core.tukey import generate_tukey


class AbstractRVSGenerator:

    @staticmethod
    def generate(size):
        raise NotImplementedError("Method is not implemented")


class BetaRVSGenerator(AbstractRVSGenerator):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def generate(self, size):
        return generate_beta(size=size, a=self.a, b=self.b)


class CauchyRVSGenerator(AbstractRVSGenerator):
    def __init__(self, t, s):
        self.t = t
        self.s = s

    def generate(self, size):
        return generate_cauchy(size=size, t=self.t, s=self.s)


class LaplaceRVSGenerator(AbstractRVSGenerator):
    def __init__(self, t, s):
        self.t = t
        self.s = s

    def generate(self, size):
        return generate_laplace(size=size, t=self.t, s=self.s)


class LogisticRVSGenerator(AbstractRVSGenerator):
    def __init__(self, t, s):
        self.t = t
        self.s = s

    def generate(self, size):
        return generate_logistic(size=size, t=self.t, s=self.s)


class TRVSGenerator(AbstractRVSGenerator):
    def __init__(self, df):
        self.df = df

    def generate(self, size):
        return generate_t(size=size, df=self.df)


class TukeyRVSGenerator(AbstractRVSGenerator):
    def __init__(self, lam):
        self.lam = lam

    def generate(self, size):
        return generate_tukey(size=size, lam=self.lam)
