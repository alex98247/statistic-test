import pytest as pytest

from stattest.test.normality import CabanaCabana2Test
from stattest.tests.AbstractTestCase import AbstractTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-0.2115733, -0.8935314, -0.1916746,  0.2805032,  1.3372893, -0.4324158,  2.8578810], 2.8578810),
    ],
)
class TestCaseCabanaCabana2Test(AbstractTestCase):

    @pytest.fixture
    def statistic_test(self):
        return CabanaCabana2Test()
