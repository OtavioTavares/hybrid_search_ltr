"""Uso: uv run search "velvet sofa" [--k 5]   (sem argumento: modo interativo)"""

import argparse

from product_search.engine import SearchEngine


def show(engine, query, k):
    try:
        out = engine.search(query, k)
    except (TypeError, ValueError) as e:
        print(f"busca inválida: {e}")
        return
    for i, r in enumerate(out["results"], 1):
        print(f"{i:>2}. [{r['product_id']}] {r['product_name']} ({r['product_class']})")
    print(f"{len(out['results'])} resultados | {out['latency_ms']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    engine = SearchEngine.load()
    if args.query is not None:
        return show(engine, args.query, args.k)
    while (query := input("busca> ")) not in {"sair", "exit"}:
        show(engine, query, args.k)


if __name__ == "__main__":
    main()
