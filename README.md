# modelbase


## Development setup

You have two choices here, using `uv` (pypi-only) or using `pixi` (conda-forge, including assimulo)

### uv

- Install `uv` as described in [the docs](https://docs.astral.sh/uv/getting-started/installation/).
- Run `uv sync --extra dev --extra torch` to install dependencies locally

### pixi

- Install `pixi` as described in [the docs](https://pixi.sh/latest/#installation)
- Run `pixi install --frozen`


## Notes

- `uv add $package`
- `uv add --optional dev $package`
