import argparse


def hello(name: str = "World") -> None:
    """Print a greeting message."""
    # TODO: Add support for multiple greeting styles (formal, casual, etc.)
    print(f"Hello, {name}!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex greeting utility")
    parser.add_argument(
        "--name",
        type=str,
        default="World",
        help="Name to greet (default: World)",
    )
    args = parser.parse_args()
    hello(args.name)


if __name__ == "__main__":
    main()
