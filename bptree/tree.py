"""On-disk B+ tree mapping ``bytes -> bytes``.

Design decisions (see README for the full write-up):

* Fixed-size pages; every node page carries a 15-byte header.
* Interior separator convention: ``max(subtree(children[i])) < keys[i]``
  ``<= min(subtree(children[i+1]))``.  Splits insert separators exactly
  (copy-up for leaves, move-up for interiors); deletes may leave a
  separator lagging behind a grown minimum — a gap in which no key can
  exist — which keeps search correct with a single cheap invariant
  (``verify`` checks both bounds).  Borrows rewrite the adjacent
  separator; merges consume it.
* Search: first ``i`` with ``key < keys[i]`` descends to ``children[i]``,
  equality descends right, otherwise the last child.
* Leaves are doubly chained through ``next_leaf`` for range scans.
* Underflow (non-root): ``used * 2 < page_size``.  Rebalance prefers a
  *comfortable* borrow (sibling stays >= half full), else a merge that
  fits, else a last-resort borrow; merges propagate up the recorded path.
"""
from __future__ import annotations

import bisect
import struct
from pathlib import Path

from .pages import (
    CHILD_SIZE,
    INTERIOR_KEY_HEAD,
    LEAF_CELL_HEAD,
    MAX_KEY,
    MAX_VALUE,
    PAGE_HEADER_SIZE,
    TYPE_FREE,
    TYPE_INTERIOR,
    TYPE_LEAF,
    Node,
    PageError,
    PageFile,
)


class BPTree:
    """Persistent B+ tree with insert/search/delete/range/verify."""

    def __init__(self, path: str | Path, page_size: int = 4096) -> None:
        if page_size < 512:
            raise ValueError("page_size must be >= 512")
        self._pf = PageFile(path, page_size)
        if self._pf.root == 0:
            root = self._pf.allocate()
            self._pf.write_node(root)
            self._pf.set_root(root.page_no)
            self._pf.sync()

    # ------------------------------------------------------------------ #
    @property
    def page_size(self) -> int:
        return self._pf.page_size

    @property
    def entry_count(self) -> int:
        return self._pf.entry_count

    def _capacity(self) -> int:
        return self.page_size

    def _leaf_cell(self, key: bytes, value: bytes) -> int:
        return LEAF_CELL_HEAD.size + len(key) + len(value)

    def _interior_cell(self, key: bytes) -> int:
        return INTERIOR_KEY_HEAD.size + len(key) + CHILD_SIZE

    @staticmethod
    def _coerce(key, value=None):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes or str")
        if value is not None and isinstance(value, str):
            value = value.encode("utf-8")
        if value is not None and not isinstance(value, bytes):
            raise TypeError("value must be bytes or str")
        return key, value

    def _check_sizes(self, key: bytes, value: bytes) -> None:
        if len(key) > MAX_KEY:
            raise ValueError(f"key longer than {MAX_KEY} bytes")
        if len(value) > MAX_VALUE:
            raise ValueError(f"value longer than {MAX_VALUE} bytes")
        if self._leaf_cell(key, value) + PAGE_HEADER_SIZE > self.page_size:
            raise ValueError("key/value pair does not fit in one page")

    @staticmethod
    def _child_index(node: Node, key: bytes) -> int:
        for i, k in enumerate(node.keys):
            if key < k:
                return i
            if key == k:
                return i + 1
        return len(node.keys)

    def _read(self, page_no: int) -> Node:
        return self._pf.read_node(page_no)

    def _descend(self, key: bytes | None):
        """Return ``(path, leaf)``; path is [(node, child_index), ...]."""
        path = []
        node = self._read(self._pf.root)
        while not node.is_leaf:
            idx = 0 if key is None else self._child_index(node, key)
            path.append((node, idx))
            node = self._read(node.children[idx])
        return path, node

    # ------------------------------------------------------------------ #
    # search
    def search(self, key: bytes) -> bytes:
        """Return the value for ``key``; raise ``KeyError`` if absent."""
        key = self._coerce(key)[0]
        _path, leaf = self._descend(key)
        pos = bisect.bisect_left(leaf.keys, key)
        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            return leaf.values[pos]
        raise KeyError(key)

    def get(self, key: bytes) -> bytes | None:
        """Return the value for ``key`` or ``None`` if absent."""
        try:
            return self.search(key)
        except KeyError:
            return None

    def __contains__(self, key: bytes) -> bool:
        return self.get(key) is not None

    # ------------------------------------------------------------------ #
    # insert
    def insert(self, key: bytes, value: bytes) -> bool:
        """Insert ``key``; overwrite if present.  True when newly created."""
        key, value = self._coerce(key, value)
        self._check_sizes(key, value)
        path, leaf = self._descend(key)
        pos = bisect.bisect_left(leaf.keys, key)
        created = not (pos < len(leaf.keys) and leaf.keys[pos] == key)
        if created:
            leaf.keys.insert(pos, key)
            leaf.values.insert(pos, value)
            self._pf.entry_count += 1
        else:
            leaf.values[pos] = value
        if leaf.used_bytes() > self.page_size:
            self._split_leaf(leaf, path)
        else:
            self._pf.write_node(leaf)
        self._pf.sync()
        return created

    def _split_leaf(self, leaf: Node, path) -> None:
        cells_total = sum(
            self._leaf_cell(k, v) for k, v in zip(leaf.keys, leaf.values)
        )
        mid = self._balanced_mid(
            [self._leaf_cell(k, v) for k, v in zip(leaf.keys, leaf.values)],
            cells_total,
        )
        right = self._pf.allocate()
        right.node_type = TYPE_LEAF
        right.keys = leaf.keys[mid:]
        right.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        right.next_leaf = leaf.next_leaf
        leaf.next_leaf = right.page_no
        self._pf.write_node(leaf)
        self._pf.write_node(right)
        self._insert_into_parent(path, leaf.page_no, right.keys[0], right.page_no)

    @staticmethod
    def _balanced_mid(sizes: list[int], total: int) -> int:
        """Split index minimizing the byte imbalance of both halves."""
        best, best_diff = 1, None
        acc = 0
        for i, s in enumerate(sizes[:-1], start=1):
            acc += s
            diff = abs(acc - (total - acc))
            if best_diff is None or diff < best_diff:
                best, best_diff = i, diff
        return best

    def _insert_into_parent(self, path, left_page: int, sep: bytes, right_page: int) -> None:
        if not path:
            root = self._pf.allocate()
            root.node_type = TYPE_INTERIOR
            root.keys = [sep]
            root.children = [left_page, right_page]
            self._pf.write_node(root)
            self._pf.set_root(root.page_no)
            return
        parent, idx = path[-1]
        parent.keys.insert(idx, sep)
        parent.children.insert(idx + 1, right_page)
        if parent.used_bytes() > self.page_size:
            self._split_interior(parent, path[:-1])
        else:
            self._pf.write_node(parent)

    def _split_interior(self, node: Node, path) -> None:
        n = len(node.keys)
        m = n // 2
        sep = node.keys[m]
        right = self._pf.allocate()
        right.node_type = TYPE_INTERIOR
        right.keys = node.keys[m + 1 :]
        right.children = node.children[m + 1 :]
        node.keys = node.keys[:m]
        node.children = node.children[: m + 1]
        self._pf.write_node(node)
        self._pf.write_node(right)
        self._insert_into_parent(path, node.page_no, sep, right.page_no)

    # ------------------------------------------------------------------ #
    # delete
    def delete(self, key: bytes) -> bool:
        """Remove ``key``.  Returns True when the key existed."""
        key = self._coerce(key)[0]
        path, leaf = self._descend(key)
        pos = bisect.bisect_left(leaf.keys, key)
        if pos >= len(leaf.keys) or leaf.keys[pos] != key:
            return False
        del leaf.keys[pos]
        del leaf.values[pos]
        self._pf.entry_count -= 1
        self._pf.write_node(leaf)
        if path and self._underflow(leaf):
            self._rebalance(path, leaf, key)
        self._pf.sync()
        return True

    def _underflow(self, node: Node) -> bool:
        return node.used_bytes() * 2 < self.page_size

    # ------------------------------------------------------------------ #
    # rebalance
    def _rebalance(self, path, node: Node, anchor: bytes) -> None:
        """Restore fill invariants bottom-up; ``anchor`` re-descends after
        every structural change so recorded positions never go stale."""
        while True:
            if not path:
                # ``node`` is the root: exempt from fill, but collapse a
                # root interior that was reduced to a single child.
                if not node.is_leaf and len(node.children) == 1:
                    only = node.children[0]
                    self._pf.free(node.page_no)
                    self._pf.set_root(only)
                return
            parent, pos = path[-1]
            if len(parent.children) == 1:
                if len(path) == 1:
                    only = parent.children[0]
                    self._pf.free(parent.page_no)
                    self._pf.set_root(only)
                    return
                # non-root parent lost its siblings' view: fix it first,
                # then recover our position by re-descending.
                self._rebalance(path[:-1], parent, anchor)
                path, node = self._recover(node, anchor)
                if not self._underflow(node):
                    return
                continue
            if not self._underflow(node):
                if parent.used_bytes() > self.page_size:
                    self._split_interior(parent, path[:-1])
                elif self._underflow(parent):
                    self._rebalance(path[:-1], parent, anchor)
                return
            left = self._read(parent.children[pos - 1]) if pos > 0 else None
            right = (
                self._read(parent.children[pos + 1])
                if pos + 1 < len(parent.children)
                else None
            )
            if left is not None and self._comfortable(left, node, from_left=True):
                self._borrow_left(parent, pos, left, node)
                path, node = self._recover(node, anchor)
                continue
            if right is not None and self._comfortable(right, node, from_left=False):
                self._borrow_right(parent, pos, node, right)
                path, node = self._recover(node, anchor)
                continue
            if left is not None and self._merge_fits(left, node, parent.keys[pos - 1]):
                self._merge_left(parent, pos, left, node)
                path, node = self._recover(left, anchor)
                continue
            if right is not None and self._merge_fits(node, right, parent.keys[pos]):
                self._merge_right(parent, pos, node, right)
                path, node = self._recover(node, anchor)
                continue
            if left is not None and len(left.keys) >= 2:
                self._borrow_left(parent, pos, left, node)
                path, node = self._recover(node, anchor)
                continue
            if right is not None and len(right.keys) >= 2:
                self._borrow_right(parent, pos, node, right)
                path, node = self._recover(node, anchor)
                continue
            raise PageError("unable to rebalance: no sibling can lend or merge")

    def _recover(self, node: Node, anchor: bytes):
        """Re-descend to the leaf that holds ``node``'s first key.

        ``node.keys[0]`` is a stored key, so routing is exact even when
        separators lag; an empty node falls back to the deleted key,
        whose old slot this node still covers.
        """
        return self._descend(node.keys[0] if node.keys else anchor)

    @staticmethod
    def _moved_size(donor: Node, from_left: bool, target_is_leaf: bool) -> int:
        if target_is_leaf:
            i = len(donor.keys) - 1 if from_left else 0
            return LEAF_CELL_HEAD.size + len(donor.keys[i]) + len(donor.values[i])
        i = len(donor.keys) - 1 if from_left else 0
        return INTERIOR_KEY_HEAD.size + len(donor.keys[i]) + CHILD_SIZE

    def _comfortable(self, donor: Node, target: Node, from_left: bool) -> bool:
        """True when lending one cell keeps the donor >= half a page."""
        if len(donor.keys) < 2:
            return False
        moved = self._moved_size(donor, from_left, target.is_leaf)
        return donor.used_bytes() - moved >= self.page_size // 2

    def _merge_fits(self, left: Node, right: Node, sep: bytes) -> bool:
        extra = 0 if left.is_leaf else INTERIOR_KEY_HEAD.size + len(sep)
        return (
            left.used_bytes() + right.used_bytes() - PAGE_HEADER_SIZE + extra
            <= self.page_size
        )

    # -- borrow / merge primitives --------------------------------------- #
    def _borrow_left(self, parent: Node, pos: int, left: Node, node: Node) -> None:
        if node.is_leaf:
            k = left.keys.pop()
            v = left.values.pop()
            node.keys.insert(0, k)
            node.values.insert(0, v)
            parent.keys[pos - 1] = k
        else:
            k = left.keys.pop()
            child = left.children.pop()
            node.keys.insert(0, k)
            node.children.insert(0, child)
            parent.keys[pos - 1] = k
        self._pf.write_node(left)
        self._pf.write_node(node)
        self._pf.write_node(parent)

    def _borrow_right(self, parent: Node, pos: int, node: Node, right: Node) -> None:
        if node.is_leaf:
            k = right.keys.pop(0)
            v = right.values.pop(0)
            node.keys.append(k)
            node.values.append(v)
            parent.keys[pos] = right.keys[0]
        else:
            k = right.keys.pop(0)
            child = right.children.pop(0)
            node.keys.append(k)
            node.children.append(child)
            parent.keys[pos] = right.keys[0]
        self._pf.write_node(right)
        self._pf.write_node(node)
        self._pf.write_node(parent)

    def _merge_left(self, parent: Node, pos: int, left: Node, node: Node) -> None:
        if node.is_leaf:
            left.keys.extend(node.keys)
            left.values.extend(node.values)
            left.next_leaf = node.next_leaf
        else:
            left.keys.append(parent.keys[pos - 1])
            left.keys.extend(node.keys)
            left.children.extend(node.children)
        del parent.keys[pos - 1]
        del parent.children[pos]
        self._pf.free(node.page_no)
        self._pf.write_node(left)
        self._pf.write_node(parent)

    def _merge_right(self, parent: Node, pos: int, node: Node, right: Node) -> None:
        if node.is_leaf:
            node.keys.extend(right.keys)
            node.values.extend(right.values)
            node.next_leaf = right.next_leaf
        else:
            node.keys.append(parent.keys[pos])
            node.keys.extend(right.keys)
            node.children.extend(right.children)
        del parent.keys[pos]
        del parent.children[pos + 1]
        self._pf.free(right.page_no)
        self._pf.write_node(node)
        self._pf.write_node(parent)

    # ------------------------------------------------------------------ #
    # range scan
    def range(self, start: bytes | None = None, end: bytes | None = None):
        """Yield ``(key, value)`` for keys in ``[start, end)`` in order."""
        if start is not None:
            start = self._coerce(start)[0]
        if end is not None:
            end = self._coerce(end)[0]
        _path, leaf = self._descend(start)
        while True:
            for k, v in zip(leaf.keys, leaf.values):
                if start is not None and k < start:
                    continue
                if end is not None and k >= end:
                    return
                yield k, v
            if leaf.next_leaf == 0:
                return
            leaf = self._read(leaf.next_leaf)

    # ------------------------------------------------------------------ #
    # stats
    def stats(self) -> dict:
        """Layout statistics: height, page counts, fill factors."""
        leaves = interiors = 0
        fills: list[float] = []

        def walk(page_no: int, depth: int) -> int:
            nonlocal leaves, interiors
            node = self._read(page_no)
            fills.append(node.used_bytes() / self.page_size)
            if node.is_leaf:
                leaves += 1
                return depth
            interiors += 1
            return max(walk(c, depth + 1) for c in node.children)

        height = walk(self._pf.root, 1) if self._pf.page_count > 1 else 0
        free_pages = 0
        page_no = self._pf.free_head
        seen = set()
        while page_no and page_no not in seen:
            seen.add(page_no)
            free_pages += 1
            buf = self._pf.read_page(page_no)
            page_no = struct.unpack_from("<I", buf, 7)[0]
        return {
            "entries": self._pf.entry_count,
            "height": height,
            "page_size": self.page_size,
            "pages": self._pf.page_count - 1,
            "free_pages": free_pages,
            "leaf_pages": leaves,
            "interior_pages": interiors,
            "min_fill": round(min(fills), 3) if fills else 0.0,
            "max_fill": round(max(fills), 3) if fills else 0.0,
            "root": self._pf.root,
        }

    # ------------------------------------------------------------------ #
    # verification
    def verify(self) -> list[str]:
        """Check all tree invariants; return a list of problems (empty = OK)."""
        problems: list[str] = []
        if self._pf.root <= 0 or self._pf.root >= self._pf.page_count:
            return [f"root page {self._pf.root} out of range"]
        leaves_in_order: list[Node] = []
        total_cells = 0

        def _bounds(page_no: int):
            """Return (min, max) of a subtree; (None, None) when empty."""
            node = self._read(page_no)
            while not node.is_leaf:
                if not node.children:
                    return None, None
                node = self._read(node.children[0])
            if not node.keys:
                return None, None
            first = node.keys[0]
            node = self._read(page_no)
            while not node.is_leaf:
                node = self._read(node.children[-1])
            if not node.keys:
                return None, None
            return first, node.keys[-1]

        def walk(page_no: int, depth: int, is_root: bool, siblings: list[Node]) -> int | None:
            nonlocal total_cells
            try:
                node = self._read(page_no)
            except (PageError, struct.error) as exc:
                problems.append(f"page {page_no}: {exc}")
                return None
            # ordering + structure
            if any(node.keys[i] >= node.keys[i + 1] for i in range(len(node.keys) - 1)):
                problems.append(f"page {page_no}: keys not strictly ascending")
            if node.is_leaf:
                if len(node.values) != len(node.keys):
                    problems.append(f"page {page_no}: value count mismatch")
                total_cells += len(node.keys)
                leaves_in_order.append(node)
                if not is_root and not node.keys:
                    problems.append(f"page {page_no}: empty non-root leaf")
            else:
                if len(node.children) != len(node.keys) + 1:
                    problems.append(f"page {page_no}: children != keys + 1")
                if not is_root and len(node.children) < 2:
                    problems.append(f"page {page_no}: interior with < 2 children")
                for i, k in enumerate(node.keys):
                    lmin, lmax = _bounds(node.children[i])
                    rmin, rmax = _bounds(node.children[i + 1])
                    if lmax is None or rmin is None:
                        problems.append(f"page {page_no}: empty subtree at key {i}")
                    elif not (lmax < k <= rmin):
                        problems.append(
                            f"page {page_no}: separator {i} outside "
                            f"(max_left={lmax!r}, key={k!r}, min_right={rmin!r})"
                        )
            # fill factor (non-root): >= half a page minus one lent cell
            if not is_root:
                max_cell = max(
                    (
                        self._leaf_cell(k, v)
                        for nd in siblings + [node]
                        for k, v in zip(nd.keys, nd.values)
                    ),
                    default=0,
                ) if node.is_leaf else max(
                    (
                        self._interior_cell(k)
                        for nd in siblings + [node]
                        for k in nd.keys
                    ),
                    default=0,
                )
                floor = self.page_size // 2 - max_cell
                if node.used_bytes() < floor:
                    problems.append(
                        f"page {page_no}: fill {node.used_bytes()} < floor {floor}"
                    )
            if node.is_leaf:
                return depth
            depths = []
            for i, child in enumerate(node.children):
                others = [
                    self._read(c)
                    for j, c in enumerate(node.children)
                    if j != i and c != child
                ]
                d = walk(child, depth + 1, False, others)
                if d is not None:
                    depths.append(d)
            if depths and len(set(depths)) > 1:
                problems.append(f"page {page_no}: leaves at different depths")
            return max(depths) if depths else None

        walk(self._pf.root, 1, True, [])

        # uniform height across the whole tree
        # (checked per interior; a single-leaf root is trivially uniform)

        # leaf chain integrity
        chain: list[Node] = []
        page_no = leaves_in_order[0].page_no if leaves_in_order else 0
        seen: set[int] = set()
        while page_no:
            if page_no in seen:
                problems.append("leaf chain contains a cycle")
                break
            seen.add(page_no)
            node = self._read(page_no)
            if not node.is_leaf:
                problems.append(f"leaf chain walks into non-leaf page {page_no}")
                break
            chain.append(node)
            page_no = node.next_leaf
        if [n.page_no for n in chain] != [n.page_no for n in leaves_in_order]:
            problems.append("leaf chain does not match in-order leaf sequence")
        for a, b in zip(chain, chain[1:]):
            if a.keys and b.keys and a.keys[-1] >= b.keys[0]:
                problems.append("leaf chain key order violated")
        if chain and chain[-1].next_leaf != 0:
            problems.append("last leaf has a dangling next_leaf pointer")

        if total_cells != self._pf.entry_count:
            problems.append(
                f"entry_count {self._pf.entry_count} != leaf cells {total_cells}"
            )

        # reachability vs free list: every page is either in the tree or free
        reachable: set[int] = set()

        def mark(page_no: int) -> None:
            if page_no in reachable:
                return
            reachable.add(page_no)
            node = self._read(page_no)
            if not node.is_leaf:
                for c in node.children:
                    mark(c)

        try:
            mark(self._pf.root)
        except PageError as exc:
            problems.append(f"reachability: {exc}")
        free_seen: set[int] = set()
        page_no = self._pf.free_head
        while page_no:
            if page_no in free_seen:
                problems.append("free list contains a cycle")
                break
            free_seen.add(page_no)
            buf = self._pf.read_page(page_no)
            if buf[0] != TYPE_FREE:
                problems.append(f"free list page {page_no} not marked FREE")
                break
            page_no = struct.unpack_from("<I", buf, 7)[0]
        overlap = reachable & free_seen
        if overlap:
            problems.append(f"pages both reachable and free: {sorted(overlap)[:5]}")
        all_pages = set(range(1, self._pf.page_count))
        orphans = all_pages - reachable - free_seen
        if orphans:
            problems.append(f"orphaned pages: {sorted(orphans)[:5]}")
        return problems

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._pf.close()

    def __enter__(self) -> "BPTree":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
