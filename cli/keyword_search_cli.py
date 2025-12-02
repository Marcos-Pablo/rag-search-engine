#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = []
            with open('./data/movies.json', 'r') as file:
                data = json.load(file)

            for movie in data["movies"]:
                if args.query in movie["title"]:
                    result.append(movie)

                if len(result) >= 5:
                    break

            result.sort(key=lambda movie: movie["title"])

            for i in range(len(result)):
                print(f"{i + 1}. {result[i]["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
