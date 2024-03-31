import pytest as pytest

from stattest.test.normality import LillieforsTest
from stattest.tests.AbstractTestCase import AbstractTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-1, 0, 1], 0.17467808),
    ],
)
class TestCaseLillieforsTestTest(AbstractTestCase):

    @pytest.fixture
    def statistic_test(self):
        return LillieforsTest()
