"""Split por query -> reports/data/splits.csv. Roda uma vez: refazer invalida tudo que veio depois.

Uso: uv run split-queries [--force]
"""

import argparse

from product_search import config
from product_search.data.loading import load_labels, load_queries
from product_search.data.split import split_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    path = config.REPORTS / "data" / "splits.csv"
    if path.exists() and not parser.parse_args().force:
        raise SystemExit(f"{path.name} já existe; use --force para refazer")

    labels = load_labels()
    splits = split_queries(labels, load_queries(labels).query_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    print(splits.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
