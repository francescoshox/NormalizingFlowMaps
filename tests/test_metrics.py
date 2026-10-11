import numpy as np

from nfm.fid.metrics import Moments, frechet


def test_streaming_fid_statistics():
    features = np.random.default_rng(2).normal(size=(73, 8))
    moments = Moments()
    for batch in np.array_split(features, 6):
        moments.update(batch)
    mean, covariance = moments.statistics()
    np.testing.assert_allclose(mean, features.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(covariance, np.cov(features, rowvar=False), atol=1e-14)
    assert frechet(mean, covariance, mean, covariance) < 1e-10
