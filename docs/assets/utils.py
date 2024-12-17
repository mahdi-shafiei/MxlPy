from __future__ import annotations


def print_as_table(items: list[str], max_len: int = 79) -> None:
    from math import floor

    from more_itertools import chunked

    longest_word = max(len(i) for i in items)
    items_per_row = floor(max_len / (longest_word + 3))
    space_per_word = floor(max_len / items_per_row)

    line_fmt = " | ".join(f"{{:<{space_per_word}}}" for i in range(items_per_row))

    for group in chunked(items, items_per_row):
        print(line_fmt.format(*group))  # noqa: T201
