import argparse

from lib.describe_image import describe_image


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--query", type=str, required=True, help="Query to describe the image"
    )

    args = parser.parse_args()
    describe_image(args.image, args.query)


if __name__ == "__main__":
    main()
