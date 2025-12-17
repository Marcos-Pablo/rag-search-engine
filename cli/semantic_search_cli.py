#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_CHUNK_SIZE
from lib.semantic_search import chunk_command, embed_query_text, embed_text_command, search_command, verify_embeddings, verify_model_command

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify that the embeddings are created correctly")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk in words"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
