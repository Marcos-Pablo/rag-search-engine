#!/usr/bin/env python3

import argparse
from lib.keyword_search import build_command, idf_command, search_command, tf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    term_frequency_parser = subparsers.add_parser("tf", help="Get term frequency for a given document ID and term")
    term_frequency_parser.add_argument("doc_id", type=int, help="Document ID")
    term_frequency_parser.add_argument("term", type=str, help="Term to get the frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built succesfully.")

        case "search":
            print(f"Searching for: {args.query}")

            result = search_command(args.query)
            for i, res in enumerate(result, 1):
                print(f"{i}. ({res['id']}) {res['title']}")

        case "tf":
            doc_id, term = args.doc_id, args.term
            print(f"Getting the term frequency for {term} in document {doc_id}")
            frequency = tf_command(doc_id, term)
            print(f"Term frequency of '{term}' in document '{doc_id}': {frequency}")

        case "idf":
            print(f"Getting the inverse document frequency of {args.term}")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
