import pytest as pytest

from stattest.test.normality import ADTest
from stattest.tests.AbstractTestCase import AbstractTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Normal with mean = 0, variance = 1
        ([16, 18, 16, 14, 12, 12, 16, 18, 16, 14, 12, 12], 0.6822883),
    ],
)
class TestCaseADTest(AbstractTestCase):

    @pytest.fixture
    def statistic_test(self):
        return ADTest()
