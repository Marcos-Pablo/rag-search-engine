#! /usr/bin/env python3

import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_parser.add_argument("image", type=str, help="Path to image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search based on the given image"
    )
    image_search_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image)
            for i, res in enumerate(results, 1):
                print()
                print(
                    f"{i}. {res['title']} (similarity: {res['similarity_score']:.3f})"
                )
                print(res["description"][:100])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
