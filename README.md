# modelbase

## Installation

You can install modelbase using pip: `pip install modelbase2`

If you want access to the sundials solver suite via the [assimulo](https://jmodelica.org/assimulo/) package, we recommend setting up a virtual environment via [pixi](https://pixi.sh/) or [mamba / conda](https://mamba.readthedocs.io/en/latest/) using the [conda-forge](https://conda-forge.org/) channel.

```bash
pixi init
pixi add python assimulo
pixi add --pypi modelbase2
```


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
