# Bough

A generic **on-disk B+ tree** mapping `bytes -> bytes`: fixed-size pages,
node splits on overflow, merge/rebalance on underflow, linked leaves for
range scans, a root pointer in the file header, and a `verify` command
that checks every tree invariant — including after adversarial workloads.

**Why it's special:** most B+ tree demos only insert. This one survives
seeded 1 200-operation random churn, ascending/descending sweeps, and a
full delete-all drain — with `verify` proving key ordering, separator
bounds, fill factors, leaf-chain integrity, uniform height, and
page-accounting (every page is reachable or on the free list) at every
step.

**Zero dependencies** — Python standard library only.

## File format

```
page 0   file header:  b"BPtree1\0" | u32 page_size | u32 root
                        | u32 free_head | u32 entry_count | u32 page_count
page n   node page:    u8 type | u16 num_cells | u32 used_bytes
                        | u32 next_leaf | u32 reserved     (15-byte header)
                        [ cells ... ]

leaf cell:      u16 klen | u16 vlen | key | value
interior cell:  u32 child0, then per cell: u16 klen | key | u32 child
```

Interior separators satisfy
`max(subtree(children[i])) < keys[i] <= min(subtree(children[i+1]))` —
splits insert separators exactly (copy-up for leaves, move-up for
interiors); deletes may leave a separator lagging behind a grown minimum,
a gap in which no key can exist, which keeps search correct with one
cheap invariant.  Freelist pages are chained through their `next_leaf`
field and never reachable from the root.

**Rebalance policy:** a non-root node is underfull when
`used * 2 < page_size`.  Rebalance prefers a *comfortable* borrow (the
donor stays ≥ half a page), else a merge that fits, else a last-resort
borrow; merges propagate up the recorded path, and a root interior
reduced to one child collapses (height shrinks).  Keys are capped at 256
bytes and values at 1 024 bytes so fill-factor floors stay meaningful.

## Runnable example

```console
> python -m bptree C:\temp\bp.db --page-size 512 put apple 1
inserted
> python -m bptree C:\temp\bp.db put banana 2
inserted
> python -m bptree C:\temp\bp.db put cherry 3
inserted
> python -m bptree C:\temp\bp.db put banana 22
overwritten
> python -m bptree C:\temp\bp.db get banana
22
> python -m bptree C:\temp\bp.db del apple
deleted
> python -m bptree C:\temp\bp.db range
banana	22
cherry	3
> python -m bptree C:\temp\bp.db verify
OK: all invariants hold
> python -m bptree C:\temp\bp.db stats
{
  "entries": 2,
  "free_pages": 0,
  "height": 1,
  "interior_pages": 0,
  "leaf_pages": 1,
  "max_fill": 0.074,
  "min_fill": 0.074,
  "page_size": 512,
  "pages": 1,
  "root": 1
}
```

Library use:

```python
from bptree import BPTree

with BPTree("C:/temp/lib.db", page_size=4096) as tree:
    tree.insert(b"key", b"value")   # True = newly created
    tree.search(b"key")             # b'value', KeyError if absent
    tree.delete(b"key")             # True when it existed
    list(tree.range(b"a", b"z"))    # sorted (key, value) pairs, [start, end)
    problems = tree.verify()        # [] means every invariant holds
```

## Test command

```
python -m unittest discover -s tests -v
```

21 tests: CRUD vs a reference dict over seeded random workloads, ordered
range scans, forced splits (height ≥ 2), underflow merges returning
pages to the free list, persistence across reopen (including the header
root pointer), adversarial sequences, `verify` catching corrupted entry
counts, out-of-range roots, severed leaf chains and tampered separators,
plus the CLI end to end.
