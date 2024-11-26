import numpy as np
import pandas as pd

from modelbase2.distributions import Beta, LogNormal, Normal, Skewnorm, Uniform, sample


def test_beta_basic_sampling() -> None:
    """Test basic Beta sampling functionality"""
    beta = Beta(a=2.0, b=3.0)
    samples = beta.sample(1000)
    assert len(samples) == 1000
    assert np.all(samples >= 0) and np.all(samples <= 1)


def test_uniform_basic_sampling() -> None:
    """Test basic Uniform sampling functionality"""
    uniform = Uniform(lower_bound=0.0, upper_bound=1.0)
    samples = uniform.sample(1000)
    assert len(samples) == 1000
    assert np.all(samples >= 0.0) and np.all(samples <= 1.0)


def test_normal_basic_sampling() -> None:
    """Test basic Normal sampling functionality"""
    normal = Normal(loc=0.0, scale=1.0)
    samples = normal.sample(1000)
    assert len(samples) == 1000
    assert np.isclose(np.mean(samples), 0.0, atol=0.1)
    assert np.isclose(np.std(samples), 1.0, atol=0.1)


def test_lognormal_basic_sampling() -> None:
    """Test basic LogNormal sampling functionality"""
    lognormal = LogNormal(mean=0.0, sigma=1.0)
    samples = lognormal.sample(1000)
    assert len(samples) == 1000
    assert np.all(samples > 0)


def test_skewnorm_basic_sampling() -> None:
    """Test basic Skewnorm sampling functionality"""
    skewnorm = Skewnorm(loc=0.0, scale=1.0, a=4.0)
    samples = skewnorm.sample(1000)
    assert len(samples) == 1000
    assert np.isclose(np.mean(samples), 0.0, atol=0.5)  # Skewed distribution


def test_sample_function() -> None:
    """Test the sample function with multiple distributions"""
    parameters = {
        "beta": Beta(a=2.0, b=3.0),
        "uniform": Uniform(lower_bound=0.0, upper_bound=1.0),
        "normal": Normal(loc=0.0, scale=1.0),
    }
    df = sample(parameters, 1000)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 3)
    assert "beta" in df.columns
    assert "uniform" in df.columns
    assert "normal" in df.columns
