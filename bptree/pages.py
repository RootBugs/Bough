"""Page file layer for the B+ tree: fixed-size pages, header, free list.

File layout::

    page 0  file header:  b"BPtree1\\0" | u32 page_size | u32 root
                           | u32 free_head | u32 entry_count | u32 page_count
    page n  node page:    u8 type | u16 num_cells | u32 used_bytes
                           | u32 next_leaf | u32 reserved   (16-byte header)
                           [ cells ... ]

Cell encodings::

    leaf:     u16 klen | u16 vlen | key | value
    interior: u32 child0, then per cell: u16 klen | key | u32 child
              separators satisfy:
              max(subtree(children[i])) < keys[i] <= min(subtree(children[i+1]))
"""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"BPtree1\x00"
FILE_HEADER_FMT = "<8sIIIII"
FILE_HEADER_SIZE = 64

PAGE_HEADER_FMT = "<BHI II"
PAGE_HEADER_SIZE = struct.calcsize(PAGE_HEADER_FMT)

TYPE_FREE = 0
TYPE_LEAF = 1
TYPE_INTERIOR = 2

MAX_KEY = 256
MAX_VALUE = 1024
LEAF_CELL_HEAD = struct.Struct("<HH")
INTERIOR_KEY_HEAD = struct.Struct("<H")
CHILD_SIZE = 4


class PageError(Exception):
    """Raised for structural page-file problems."""


class Node:
    """Decoded leaf or interior page."""

    def __init__(self, page_no: int, node_type: int) -> None:
        self.page_no = page_no
        self.node_type = node_type
        self.next_leaf = 0
        self.keys: list[bytes] = []
        self.values: list[bytes] = []  # leaf only
        self.children: list[int] = []  # interior only

    @property
    def is_leaf(self) -> bool:
        return self.node_type == TYPE_LEAF

    # ------------------------------------------------------------------ #
    def used_bytes(self) -> int:
        if self.is_leaf:
            used = PAGE_HEADER_SIZE
            for k, v in zip(self.keys, self.values):
                used += LEAF_CELL_HEAD.size + len(k) + len(v)
            return used
        used = PAGE_HEADER_SIZE + CHILD_SIZE
        for k in self.keys:
            used += INTERIOR_KEY_HEAD.size + len(k) + CHILD_SIZE
        return used

    @classmethod
    def decode(cls, page_no: int, data: bytes, page_size: int) -> "Node":
        if len(data) != page_size:
            raise PageError(f"page {page_no}: size {len(data)} != {page_size}")
        ptype, num, used, nxt, _res = struct.unpack_from(PAGE_HEADER_FMT, data, 0)
        if ptype not in (TYPE_LEAF, TYPE_INTERIOR):
            raise PageError(f"page {page_no}: bad page type {ptype}")
        node = cls(page_no, ptype)
        node.next_leaf = nxt
        pos = PAGE_HEADER_SIZE
        if ptype == TYPE_LEAF:
            for _ in range(num):
                klen, vlen = LEAF_CELL_HEAD.unpack_from(data, pos)
                pos += LEAF_CELL_HEAD.size
                if pos + klen + vlen > page_size:
                    raise PageError(f"page {page_no}: leaf cell overruns page")
                node.keys.append(bytes(data[pos : pos + klen]))
                pos += klen
                node.values.append(bytes(data[pos : pos + vlen]))
                pos += vlen
        else:
            (child0,) = struct.unpack_from("<I", data, pos)
            pos += CHILD_SIZE
            node.children.append(child0)
            for _ in range(num):
                (klen,) = INTERIOR_KEY_HEAD.unpack_from(data, pos)
                pos += INTERIOR_KEY_HEAD.size
                if pos + klen + CHILD_SIZE > page_size:
                    raise PageError(f"page {page_no}: interior cell overruns page")
                node.keys.append(bytes(data[pos : pos + klen]))
                pos += klen
                (child,) = struct.unpack_from("<I", data, pos)
                pos += CHILD_SIZE
                node.children.append(child)
        if used != page_size and used < pos:
            raise PageError(f"page {page_no}: used_bytes {used} < parsed {pos}")
        return node

    def encode(self, page_size: int) -> bytes:
        used = self.used_bytes()
        if used > page_size:
            raise PageError(f"page {self.page_no}: node does not fit ({used}>{page_size})")
        out = bytearray(page_size)
        num = len(self.keys)
        struct.pack_into(
            PAGE_HEADER_FMT, out, 0, self.node_type, num, used, self.next_leaf, 0
        )
        pos = PAGE_HEADER_SIZE
        if self.is_leaf:
            for k, v in zip(self.keys, self.values):
                LEAF_CELL_HEAD.pack_into(out, pos, len(k), len(v))
                pos += LEAF_CELL_HEAD.size
                out[pos : pos + len(k)] = k
                pos += len(k)
                out[pos : pos + len(v)] = v
                pos += len(v)
        else:
            struct.pack_into("<I", out, pos, self.children[0])
            pos += CHILD_SIZE
            for k, child in zip(self.keys, self.children[1:]):
                INTERIOR_KEY_HEAD.pack_into(out, pos, len(k))
                pos += INTERIOR_KEY_HEAD.size
                out[pos : pos + len(k)] = k
                pos += len(k)
                struct.pack_into("<I", out, pos, child)
                pos += CHILD_SIZE
        return bytes(out)


class PageFile:
    """Fixed-size page allocator with a free list and write-through cache."""

    def __init__(self, path: str | Path, page_size: int = 4096) -> None:
        if page_size < 256:
            raise ValueError("page_size must be >= 256")
        self.path = Path(path)
        self._cache: dict[int, bytearray] = {}
        if not self.path.exists():
            self.page_size = page_size
            self.root = 0
            self.free_head = 0
            self.entry_count = 0
            self.page_count = 1
            self._fh = self.path.open("w+b")
            self._fh.truncate(page_size)
            self._write_header()
        else:
            self._fh = self.path.open("r+b")
            self._read_header()
            if self.page_size != page_size:
                page_size = self.page_size  # honor stored size

    # ------------------------------------------------------------------ #
    def _read_header(self) -> None:
        self._fh.seek(0)
        blob = self._fh.read(FILE_HEADER_SIZE)
        if len(blob) < FILE_HEADER_SIZE:
            raise PageError("file too small to be a B+ tree file")
        magic, page_size, root, free_head, count, pages = struct.unpack_from(
            FILE_HEADER_FMT, blob, 0
        )
        if magic != MAGIC:
            raise PageError("bad magic: not a B+ tree file")
        if page_size < 256:
            raise PageError("corrupt page size in header")
        self.page_size = page_size
        self.root = root
        self.free_head = free_head
        self.entry_count = count
        self.page_count = pages

    def _write_header(self) -> None:
        blob = bytearray(FILE_HEADER_SIZE)
        struct.pack_into(
            FILE_HEADER_FMT,
            blob,
            0,
            MAGIC,
            self.page_size,
            self.root,
            self.free_head,
            self.entry_count,
            self.page_count,
        )
        self._fh.seek(0)
        self._fh.write(bytes(blob))
        self._fh.flush()

    def _sync_header(self) -> None:
        self._write_header()

    # ------------------------------------------------------------------ #
    def read_node(self, page_no: int) -> Node:
        if page_no <= 0 or page_no >= self.page_count:
            raise PageError(f"page {page_no} out of range")
        data = self.read_page(page_no)
        return Node.decode(page_no, bytes(data), self.page_size)

    def read_page(self, page_no: int) -> bytearray:
        if page_no in self._cache:
            return self._cache[page_no]
        if page_no <= 0 or page_no >= self.page_count:
            raise PageError(f"page {page_no} out of range")
        self._fh.seek(page_no * self.page_size)
        blob = self._fh.read(self.page_size)
        if len(blob) != self.page_size:
            raise PageError(f"page {page_no}: short read")
        buf = bytearray(blob)
        self._cache[page_no] = buf
        return buf

    def write_node(self, node: Node) -> None:
        data = node.encode(self.page_size)
        page_no = node.page_no
        self._cache[page_no] = bytearray(data)
        self._fh.seek(page_no * self.page_size)
        self._fh.write(data)
        self._fh.flush()

    # ------------------------------------------------------------------ #
    def allocate(self) -> Node:
        """Allocate a page (free list first) and return an empty node on it."""
        if self.free_head:
            page_no = self.free_head
            buf = self.read_page(page_no)
            _ptype, _num, _used, nxt, _res = struct.unpack_from(PAGE_HEADER_FMT, buf, 0)
            self.free_head = nxt
        else:
            page_no = self.page_count
            self.page_count += 1
            self._fh.seek(0)
            self._fh.truncate(self.page_count * self.page_size)
        self._sync_header()
        return Node(page_no, TYPE_LEAF)

    def free(self, page_no: int) -> None:
        if page_no <= 0:
            raise PageError("cannot free page 0")
        buf = bytearray(self.page_size)
        struct.pack_into(PAGE_HEADER_FMT, buf, 0, TYPE_FREE, 0, 0, self.free_head, 0)
        self._cache[page_no] = buf
        self._fh.seek(page_no * self.page_size)
        self._fh.write(bytes(buf))
        self._fh.flush()
        self.free_head = page_no
        self._sync_header()

    def set_root(self, page_no: int) -> None:
        self.root = page_no
        self._sync_header()

    def sync(self) -> None:
        """Persist the file header (entry count, root, free list)."""
        self._sync_header()

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if not self._fh.closed:
            self._sync_header()
            self._fh.close()

    def __enter__(self) -> "PageFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
