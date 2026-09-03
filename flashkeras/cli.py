import argparse
import sys

from flashkeras.notebooks import describe, list_notebooks, new_notebook


def _cmd_notebooks_list(args: argparse.Namespace) -> None:
    """Print the available notebook templates, optionally filtered by tag.

    `Flash Explanation:` *`Use this internal command handler to display the notebook catalog.`*
    """
    items = list_notebooks(tag=args.tag)

    if not items:
        if args.tag:
            print(f"No notebooks found with tag '{args.tag}'.")
        else:
            print("No notebooks available.")
        return

    print(
        f"Available notebooks"
        f"{f' (tag: {args.tag})' if args.tag else ''}:\n"
    )

    for name, meta in items.items() if isinstance(items, dict) else _as_named(items):
        print(f"  {name}")
        print(f"      {meta['title']}")
        print(f"      {meta['description']}")
        print(f"      tags: {', '.join(meta['tags'])}\n")


def _as_named(items):
    """Yield notebook metadata as ``(name, metadata)`` pairs.

    `Flash Explanation:` *`Normalize catalog entries for CLI display.`*
    """
    for meta in items:
        yield meta.get("name", meta["title"]), meta


def _cmd_notebooks_describe(args: argparse.Namespace) -> None:
    """Print details for one named notebook template.

    `Flash Explanation:` *`Use this internal command handler to inspect a template from the CLI.`*
    """
    try:
        meta = describe(args.name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(args.name)
    print(f"  title:       {meta['title']}")
    print(f"  description: {meta['description']}")
    print(f"  tags:        {', '.join(meta['tags'])}")
    print(f"  filename:    {meta['filename']}")


def _cmd_notebooks_new(args: argparse.Namespace) -> None:
    """Create a notebook template at the requested destination.

    `Flash Explanation:` *`Use this internal command handler to generate a starter notebook.`*
    """
    try:
        path = new_notebook(
            args.name,
            dest=args.dest,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Created: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the flashkeras command-line argument parser.

    `Flash Explanation:` *`Use this to obtain the parser that defines notebook commands.`*
    """
    parser = argparse.ArgumentParser(
        prog="flashkeras",
        description="flashkeras CLI - helpers for fast Keras / ML pipelines.",
    )

    subparsers = parser.add_subparsers(dest="command")

    notebooks_parser = subparsers.add_parser(
        "notebooks",
        help="Browse and generate flashkeras notebook templates.",
    )

    notebooks_subparsers = notebooks_parser.add_subparsers(
        dest="notebooks_command"
    )

    list_parser = notebooks_subparsers.add_parser(
        "list",
        help="List available notebooks.",
    )
    list_parser.add_argument(
        "--tag",
        default=None,
        help="Filter notebooks by tag.",
    )
    list_parser.set_defaults(func=_cmd_notebooks_list)

    describe_parser = notebooks_subparsers.add_parser(
        "describe",
        help="Show details about a specific notebook.",
    )
    describe_parser.add_argument(
        "name",
        help="Notebook name (see 'notebooks list').",
    )
    describe_parser.set_defaults(func=_cmd_notebooks_describe)

    new_parser = notebooks_subparsers.add_parser(
        "new",
        help="Copy a notebook template to your project.",
    )
    new_parser.add_argument(
        "name",
        help="Notebook name (see 'notebooks list').",
    )
    new_parser.add_argument(
        "--dest",
        default=".",
        help="Destination path or directory (default: current directory).",
    )
    new_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    new_parser.set_defaults(func=_cmd_notebooks_new)

    return parser


def main() -> None:
    """Parse command-line arguments and dispatch the selected command.

    `Flash Explanation:` *`Use this as the executable entry point for the flashkeras CLI.`*
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "notebooks" and getattr(args, "notebooks_command", None):
        args.func(args)
    elif args.command == "notebooks":
        parser.parse_args(["notebooks", "--help"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()