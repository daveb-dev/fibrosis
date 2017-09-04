from ...nsestimator import NSEstimator
import numpy as np


def test_nsestimator_parse_parameters():
    est = NSEstimator('test_nsestimator.yaml', 'test_nsestimator.yaml')
    est._parse_parameters()

    x01 = np.array([30., -4.32192809, 0.27327185])
    xfun1 = [0, 1, 2]
    bounds1 = [None, None, [100, 3000]]

    np.testing.assert_almost_equal(est._x0, x01)
    assert est._xfun == xfun1
    assert est._bounds == bounds1

    est.options['estimation']['parameters']['inflow']['dR']['use_slip'] = 0
    est._parse_parameters()
    x02 = np.array([30., -3.32192809, -4.32192809, 0.27327185])
    xfun2 = [0, 1, 1, 2]
    bounds2 = [None, None, None, [100, 3000]]

    np.testing.assert_almost_equal(est._x0, x02)
    assert est._xfun == xfun2
    assert est._bounds == bounds2
