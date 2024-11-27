from modelbase2.distributions import Beta, LogNormal, Normal, Skewnorm, Uniform, sample


def test_sample() -> None:
    samples = sample(
        {
            "beta": Beta(a=1.0, b=1.0),
            "uniform": Uniform(lower_bound=0.0, upper_bound=1.0),
            "normal": Normal(loc=1.0, scale=0.1),
            "log_normal": LogNormal(mean=1.0, sigma=0.1),
            "skewnorm": Skewnorm(loc=1.0, scale=0.1, a=5.0),
        },
        n=1,
    )
    assert len(samples.index) == 1
    assert len(samples.columns) == 5
