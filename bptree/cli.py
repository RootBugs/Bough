"""CLI for the on-disk B+ tree.

Usage::

    python -m bptree <file> put <key> <value>
    python -m bptree <file> get <key>
    python -m bptree <file> del <key>
    python -m bptree <file> range [start] [end]
    python -m bptree <file> verify
    python -m bptree <file> stats
"""
from __future__ import annotations

import argparse
import json
import sys

from .tree import BPTree


def _show(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return repr(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bptree", description="On-disk B+ tree CLI")
    parser.add_argument("file", help="tree file (created if missing)")
    parser.add_argument("--page-size", type=int, default=4096, help="page size for new files")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("put", help="insert or overwrite a key")
    p.add_argument("key")
    p.add_argument("value")

    p = sub.add_parser("get", help="fetch a key")
    p.add_argument("key")

    p = sub.add_parser("del", help="delete a key")
    p.add_argument("key")

    p = sub.add_parser("range", help="scan keys in order")
    p.add_argument("start", nargs="?", default=None)
    p.add_argument("end", nargs="?", default=None)

    sub.add_parser("verify", help="check tree invariants")
    sub.add_parser("stats", help="print layout statistics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tree = BPTree(args.file, page_size=args.page_size)
    try:
        if args.command == "put":
            created = tree.insert(args.key, args.value)
            print("inserted" if created else "overwritten")
        elif args.command == "get":
            value = tree.get(args.key)
            if value is None:
                print("not found", file=sys.stderr)
                return 1
            print(_show(value))
        elif args.command == "del":
            if tree.delete(args.key):
                print("deleted")
            else:
                print("not found", file=sys.stderr)
                return 1
        elif args.command == "range":
            for key, value in tree.range(args.start, args.end):
                print(f"{_show(key)}\t{_show(value)}")
        elif args.command == "verify":
            problems = tree.verify()
            if problems:
                for p in problems:
                    print(f"FAIL: {p}", file=sys.stderr)
                return 1
            print("OK: all invariants hold")
        elif args.command == "stats":
            print(json.dumps(tree.stats(), indent=2, sort_keys=True))
    finally:
        tree.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
