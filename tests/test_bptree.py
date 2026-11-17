"""Core B+ tree behaviors: CRUD vs reference, ranges, splits, persistence."""
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bptree import BPTree, PageError  # noqa: E402


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bptree-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def make(self, name="tree.db", page_size=512) -> BPTree:
        return BPTree(self.dir / name, page_size=page_size)


class TestBasicOps(TreeTestCase):
    def test_insert_search_delete_roundtrip(self):
        t = self.make()
        self.assertTrue(t.insert(b"apple", b"red"))
        self.assertTrue(t.insert(b"banana", b"yellow"))
        self.assertEqual(t.search(b"apple"), b"red")
        self.assertEqual(t.get(b"banana"), b"yellow")
        self.assertIsNone(t.get(b"cherry"))
        self.assertTrue(t.delete(b"apple"))
        self.assertFalse(t.delete(b"apple"))
        with self.assertRaises(KeyError):
            t.search(b"apple")
        self.assertEqual(t.entry_count, 1)
        t.close()

    def test_overwrite_updates_value_without_growing(self):
        t = self.make()
        t.insert("k", "v1")
        self.assertFalse(t.insert("k", "v2"))  # not created
        self.assertEqual(t.get("k"), b"v2")
        self.assertEqual(t.entry_count, 1)
        t.close()

    def test_empty_tree_semantics(self):
        t = self.make()
        self.assertEqual(t.verify(), [])
        self.assertIsNone(t.get(b"nothing"))
        self.assertEqual(list(t.range()), [])
        self.assertFalse(t.delete(b"nothing"))
        self.assertEqual(t.entry_count, 0)
        t.close()

    def test_size_limits_enforced(self):
        t = self.make()
        with self.assertRaises(ValueError):
            t.insert(b"x" * 300, b"v")
        with self.assertRaises(ValueError):
            t.insert(b"k", b"v" * 5000)
        with self.assertRaises(ValueError):
            t.insert(b"k", b"ok-but-page-too-small-for-it" * 60)
        with self.assertRaises(ValueError):
            BPTree(self.dir / "small.db", page_size=128)
        t.close()


class TestWorkloads(TreeTestCase):
    def test_random_workload_matches_reference_dict(self):
        t = self.make(page_size=512)
        ref: dict[bytes, bytes] = {}
        rng = random.Random(42)
        for i in range(1200):
            op = rng.random()
            key = f"k{rng.randrange(300):03d}".encode()
            if op < 0.55:
                value = f"v{i}".encode()
                self.assertEqual(t.insert(key, value), key not in ref)
                ref[key] = value
            elif op < 0.85:
                self.assertEqual(t.delete(key), key in ref, f"op {i}")
                ref.pop(key, None)
            else:
                self.assertEqual(t.get(key), ref.get(key), f"op {i}")
        self.assertEqual(t.verify(), [])
        self.assertEqual(t.entry_count, len(ref))
        self.assertEqual([k for k, _ in t.range()], sorted(ref))
        t.close()

    def test_range_scans_match_sorted_reference(self):
        t = self.make(page_size=512)
        keys = [f"key{i:04d}".encode() for i in range(200)]
        rng = random.Random(3)
        rng.shuffle(keys)
        for k in keys:
            t.insert(k, k.upper())
        self.assertEqual([k for k, _ in t.range()], sorted(keys))
        self.assertEqual(
            [k for k, _ in t.range(b"key0050", b"key0060")],
            [f"key{i:04d}".encode() for i in range(50, 60)],
        )
        self.assertEqual(
            [k for k, _ in t.range(b"key0190", None)],
            [f"key{i:04d}".encode() for i in range(190, 200)],
        )
        self.assertEqual(
            [k for k, _ in t.range(None, b"key0003")],
            [f"key{i:04d}".encode() for i in range(3)],
        )
        t.close()

    def test_forced_splits_build_multi_level_tree(self):
        t = self.make(page_size=512)
        for i in range(400):
            t.insert(f"k{i:04d}".encode(), f"v{i}".encode())
        stats = t.stats()
        self.assertGreaterEqual(stats["height"], 2, stats)
        self.assertGreater(stats["interior_pages"], 0)
        self.assertEqual(t.verify(), [])
        for i in range(400):
            self.assertEqual(t.get(f"k{i:04d}".encode()), f"v{i}".encode())
        t.close()

    def test_underflow_merges_recover_pages(self):
        t = self.make(page_size=512)
        for i in range(400):
            t.insert(f"k{i:04d}".encode(), b"v")
        before = t.stats()
        self.assertGreater(before["pages"], 10)
        self.assertGreater(before["leaf_pages"], 5)
        for i in range(395):
            self.assertTrue(t.delete(f"k{i:04d}".encode()))
        after = t.stats()
        self.assertGreater(after["free_pages"], 0, "merged pages not freed")
        self.assertEqual(t.verify(), [])
        remaining = [k for k, _ in t.range()]
        self.assertEqual(remaining, [f"k{i:04d}".encode() for i in range(395, 400)])
        t.close()

    def test_delete_all_then_reuse(self):
        t = self.make(page_size=512)
        for i in range(200):
            t.insert(f"k{i:03d}".encode(), b"v")
        for i in range(200):
            self.assertTrue(t.delete(f"k{i:03d}".encode()))
        self.assertEqual(t.entry_count, 0)
        self.assertEqual(t.verify(), [])
        self.assertEqual(t.stats()["height"], 1, "tree should collapse back to root leaf")
        # reuse after full drain
        for i in range(50):
            t.insert(f"n{i:03d}".encode(), b"w")
        self.assertEqual(t.verify(), [])
        self.assertEqual(len(list(t.range())), 50)
        t.close()

    def test_adversarial_sequences_keep_invariants(self):
        t = self.make(page_size=512)
        for i in range(500):  # ascending
            t.insert(f"key{i:04d}".encode(), b"v")
        self.assertEqual(t.verify(), [])
        for i in range(0, 500, 2):  # delete every other
            self.assertTrue(t.delete(f"key{i:04d}".encode()))
        self.assertEqual(t.verify(), [])
        for i in range(499, -1, -1):  # drain descending
            if i % 2:
                self.assertTrue(t.delete(f"key{i:04d}".encode()))
        self.assertEqual(t.verify(), [])
        self.assertEqual(t.entry_count, 0)
        t.close()


class TestPersistence(TreeTestCase):
    def test_persisted_across_reopen(self):
        path = self.dir / "persist.db"
        t = BPTree(path, page_size=512)
        expected = {}
        for i in range(300):
            k, v = f"k{i:04d}".encode(), f"v{i}".encode()
            t.insert(k, v)
            expected[k] = v
        t.delete(b"k0000")
        del expected[b"k0000"]
        t.close()

        t2 = BPTree(path, page_size=512)
        self.assertEqual(t2.entry_count, len(expected))
        self.assertEqual(t2.verify(), [])
        for k, v in expected.items():
            self.assertEqual(t2.get(k), v)
        # header root pointer survives in the file itself
        raw = path.read_bytes()
        root_in_header = struct.unpack_from("<I", raw, 12)[0]
        self.assertEqual(root_in_header, t2.stats()["root"])
        t2.close()

    def test_reopen_continues_where_closed(self):
        path = self.dir / "cont.db"
        t = BPTree(path, page_size=512)
        for i in range(100):
            t.insert(f"a{i:03d}".encode(), b"1")
        t.close()
        t = BPTree(path, page_size=512)
        for i in range(100, 200):
            t.insert(f"a{i:03d}".encode(), b"1")
        self.assertEqual(t.entry_count, 200)
        self.assertEqual(t.verify(), [])
        t.close()


if __name__ == "__main__":
    unittest.main()
