import argparse

from lib.hybrid_search import (
    normalize_scores_command,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            result = rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
            print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
            for i, res in enumerate(result, 1):
                print(f"{i}. {res['title']}")
                if args.rerank_method == "individual":
                    print(f"   Rerank score: {res.get('individual_score', 0):.3f}/10")
                if args.rerank_method == "batch":
                    print(f"   Rerank rank: {res.get('batch_score', 0)}")
                if args.rerank_method == "cross_encoder":
                    print(f"   Rerank score: {res.get('cross_encoder_score', 0):.3f}")
                print(f"   RRF Score: {res.get('rrf_score', 0):.3f}")
                print(
                    f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                )
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
