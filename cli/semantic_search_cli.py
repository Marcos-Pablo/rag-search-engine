#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model_command

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
