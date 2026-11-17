"""Invariant verification (incl. corruption detection) and CLI tests."""
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bptree import BPTree  # noqa: E402


def run_cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "bptree", *args],
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


class VerifyTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bpverify-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "v.db"
        t = BPTree(self.path, page_size=512)
        for i in range(400):
            t.insert(f"k{i:04d}".encode(), f"v{i}".encode())
        t.close()

    def test_clean_tree_verifies(self):
        t = BPTree(self.path, page_size=512)
        self.assertEqual(t.verify(), [])
        t.close()

    def test_verify_detects_entry_count_corruption(self):
        raw = bytearray(self.path.read_bytes())
        (count,) = struct.unpack_from("<I", raw, 20)
        struct.pack_into("<I", raw, 20, count + 5)
        self.path.write_bytes(bytes(raw))
        t = BPTree(self.path, page_size=512)
        problems = t.verify()
        self.assertTrue(any("entry_count" in p for p in problems), problems)
        t.close()

    def test_verify_detects_out_of_range_root(self):
        raw = bytearray(self.path.read_bytes())
        struct.pack_into("<I", raw, 12, 99999)
        self.path.write_bytes(bytes(raw))
        t = BPTree(self.path, page_size=512)
        problems = t.verify()
        self.assertEqual(len(problems), 1)
        self.assertIn("root page", problems[0])
        t.close()

    def test_verify_detects_broken_leaf_chain(self):
        raw = bytearray(self.path.read_bytes())
        page_size = struct.unpack_from("<I", raw, 8)[0]
        root = struct.unpack_from("<I", raw, 12)[0]
        # walk to the leftmost leaf
        page_no = root
        while True:
            base = page_no * page_size
            ptype = raw[base]
            if ptype == 1:  # leaf
                break
            (page_no,) = struct.unpack_from("<I", raw, base + 15)  # child0 after header
        # sever the chain: first leaf no longer points at the second
        struct.pack_into("<I", raw, page_no * page_size + 7, 0)
        self.path.write_bytes(bytes(raw))
        t = BPTree(self.path, page_size=512)
        problems = t.verify()
        self.assertTrue(any("leaf chain" in p for p in problems), problems)
        t.close()

    def test_verify_detects_tampered_key_order(self):
        raw = bytearray(self.path.read_bytes())
        page_size = struct.unpack_from("<I", raw, 8)[0]
        root = struct.unpack_from("<I", raw, 12)[0]
        # root interior: swap two separator keys to break ordering
        base = root * page_size
        self.assertEqual(raw[base], 2, "expected interior root")
        klen0 = struct.unpack_from("<H", raw, base + 15 + 4)[0]
        off1 = base + 15 + 4 + 2 + klen0 + 4  # second cell header
        klen1 = struct.unpack_from("<H", raw, off1)[0]
        key0 = bytes(raw[base + 15 + 4 + 2 : base + 15 + 4 + 2 + klen0])
        key1 = bytes(raw[off1 + 2 : off1 + 2 + klen1])
        self.assertGreater(klen0, 0)
        # overwrite separator 0 with separator 1's bytes if it fits, else zero out
        if klen1 <= klen0:
            raw[base + 15 + 4 + 2 : base + 15 + 4 + 2 + klen1] = key1
            raw[base + 15 + 4 + 2 + klen1 : base + 15 + 4 + 2 + klen0] = b"\x00" * (
                klen0 - klen1
            )
            struct.pack_into("<H", raw, base + 15 + 4, klen1)
        self.path.write_bytes(bytes(raw))
        t = BPTree(self.path, page_size=512)
        problems = t.verify()
        self.assertTrue(
            any("separator" in p or "ascending" in p for p in problems), problems
        )
        t.close()


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bpcli-")
        self.addCleanup(self._tmp.cleanup)
        self.file = str(Path(self._tmp.name) / "cli.db")

    def test_put_get_del_flow(self):
        out = run_cli(self.file, "--page-size", "512", "put", "greeting", "hello")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "inserted")
        again = run_cli(self.file, "--page-size", "512", "put", "greeting", "hi")
        self.assertEqual(again.stdout.strip(), "overwritten")
        got = run_cli(self.file, "get", "greeting")
        self.assertEqual(got.stdout.strip(), "hi")
        missing = run_cli(self.file, "get", "nope")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("not found", missing.stderr)
        deleted = run_cli(self.file, "del", "greeting")
        self.assertEqual(deleted.stdout.strip(), "deleted")
        gone = run_cli(self.file, "del", "greeting")
        self.assertEqual(gone.returncode, 1)

    def test_range_and_stats(self):
        for k, v in [("b", "2"), ("a", "1"), ("c", "3")]:
            run_cli(self.file, "--page-size", "512", "put", k, v)
        ranged = run_cli(self.file, "range")
        self.assertEqual(ranged.returncode, 0, ranged.stderr)
        self.assertEqual(
            [ln.split("\t") for ln in ranged.stdout.splitlines()],
            [["a", "1"], ["b", "2"], ["c", "3"]],
        )
        stats = run_cli(self.file, "stats")
        self.assertEqual(stats.returncode, 0, stats.stderr)
        import json

        payload = json.loads(stats.stdout)
        self.assertEqual(payload["entries"], 3)
        self.assertGreaterEqual(payload["page_size"], 512)

    def test_verify_command(self):
        for i in range(100):
            run_cli(self.file, "--page-size", "512", "put", f"k{i}", str(i))
        ok = run_cli(self.file, "verify")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("OK", ok.stdout)

    def test_verify_command_fails_on_corruption(self):
        run_cli(self.file, "--page-size", "512", "put", "k", "v")
        raw = bytearray(Path(self.file).read_bytes())
        struct.pack_into("<I", raw, 20, 12345)
        Path(self.file).write_bytes(bytes(raw))
        bad = run_cli(self.file, "verify")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("FAIL", bad.stderr)


if __name__ == "__main__":
    unittest.main()
