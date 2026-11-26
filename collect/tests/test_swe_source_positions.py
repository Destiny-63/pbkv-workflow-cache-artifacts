"""Actual Git snapshots with isolated synthetic source; no benchmark answers."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from collect.swe_core import Transcript
from collect.swe_ma import generation_limits, sampling_params
from collect.swe_source_positions import compile_positions, registry_digest, resolve_position_edits, source_positions


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.original = "outside\nsame\nthird\nsame\nlast\n"
        (self.repo / "a.py").write_text(self.original)
        (self.repo / "b.py").write_text("one\ntwo\n")
        (self.repo / "link.py").symlink_to("a.py")
        (self.repo / "no_newline.py").write_text("abc")
        (self.repo / "crlf.py").write_bytes(b"first\r\nsecond\r\nlast\r\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "isolated synthetic fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.context, self.schema, self.prov = source_positions(self.repo, self.commit,
            [dict(path="a.py", start=1, end=4), dict(path="a.py", start=3, end=5), dict(path="b.py", start=0, end=2)])
        self.ids = list(self.prov["spans"])

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.DEVNULL)

    def edit(self, first=2, last=2, replace="changed\n", index=0):
        return dict(span_id=self.ids[index], start_line=first, end_line=last, replace=replace)

    def compile(self, edits, prov=None, commit=None):
        return compile_positions(json.dumps({"edits": edits}), self.repo, commit or self.commit, prov or self.prov)

    def applied(self, diff):
        subprocess.run(["git", "-C", str(self.repo), "apply", "--check", "-"], input=diff, text=True, capture_output=True, check=True)
        subprocess.run(["git", "-C", str(self.repo), "apply", "-"], input=diff, text=True, capture_output=True, check=True)
        return (self.repo / "a.py").read_text()

    def test_nonunique_selection_and_outside_preserved(self):
        diff, resolved = self.compile([self.edit()])
        self.assertEqual(resolved["edits"][0]["search"], "same\n")
        self.assertEqual(self.applied(diff), "outside\nchanged\nthird\nsame\nlast\n")

    def test_multiple_original_coordinates_and_input_order(self):
        edits = [self.edit(2, 2, "long\naddition\n"), self.edit(4, 4, "end\n")]
        # Canonical diff must be independent of submission order; resolved records preserve it.
        self.assertEqual(self.compile(edits)[0], self.compile(edits[::-1])[0])
        self.assertEqual(self.applied(self.compile(edits)[0]), "outside\nlong\naddition\nthird\nend\nlast\n")

    def test_deletion_and_multiple_files(self):
        diff, _ = self.compile([self.edit(2, 3, ""), self.edit(1, 2, "other\n", 2)])
        self.assertEqual(self.applied(diff), "outside\nsame\nlast\n")
        self.assertEqual((self.repo / "b.py").read_text(), "other\n")

    def test_range_types_and_bounds_rejected(self):
        for first, last in [(1, 2), (2, 5), (4, 3), (0, 0), (-1, 2), (2.0, 2), (True, 2), (2, False), (2, "3")]:
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                self.compile([self.edit(first, last)])

    def test_unknown_id_noop_newline_and_extra_fields_rejected(self):
        for changes in [dict(span_id="unknown"), dict(span_id=1), dict(replace=None), dict(replace="same\n"),
                        dict(replace="unterminated"), dict(path="b.py")]:
            edit = self.edit(); edit.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.compile([edit])

    def test_overlap_even_across_different_spans_rejected(self):
        with self.assertRaises(ValueError): self.compile([self.edit(3, 4), self.edit(4, 4, "other\n", 1)])
        with self.assertRaises(ValueError): self.compile([self.edit(), self.edit()])

    def test_wrong_commit_and_corrupt_registry_rejected(self):
        with self.assertRaises(ValueError): self.compile([self.edit()], commit="other")
        for key, value in [("base_commit", "other"), ("rendered_context_sha256", "other")]:
            prov = copy.deepcopy(self.prov); prov[key] = value
            with self.assertRaises(ValueError): self.compile([self.edit()], prov=prov)

    def test_base_and_visible_hash_checks(self):
        for changed in ["base", "visible"]:
            prov = copy.deepcopy(self.prov)
            if changed == "base": prov["base_file_sha256"]["a.py"] = "bad"
            else: prov["spans"][self.ids[0]]["source_sha256"] = "bad"
            prov["registry_sha256"] = registry_digest(prov)
            with self.subTest(changed=changed), self.assertRaises(ValueError): self.compile([self.edit()], prov=prov)

    def test_symlinks_untracked_and_traversal_rejected(self):
        for path in ["link.py", "missing.py", "../a.py", "/a.py", ".git/config"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                source_positions(self.repo, self.commit, [dict(path=path, start=0, end=1)])

    def test_no_final_newline_file_rejected_on_compile(self):
        _, _, prov = source_positions(self.repo, self.commit, [dict(path="no_newline.py", start=0, end=1)])
        edit = dict(span_id=next(iter(prov["spans"])), start_line=1, end_line=1, replace="changed\n")
        with self.assertRaises(ValueError): self.compile([edit], prov=prov)

    def test_crlf_bytes_outside_selection_preserved(self):
        context, _, prov = source_positions(self.repo, self.commit, [dict(path="crlf.py", start=1, end=2)])
        self.assertIn("000002 | second\r\n", context)
        edit = dict(span_id=next(iter(prov["spans"])), start_line=2, end_line=2, replace="changed\r\n")
        diff, _ = self.compile([edit], prov=prov)
        subprocess.run(["git", "-C", str(self.repo), "apply", "-"], input=diff.encode(), capture_output=True, check=True)
        self.assertEqual((self.repo / "crlf.py").read_bytes(), b"first\r\nchanged\r\nlast\r\n")

    def test_registration_bounds_and_duplicate_ids(self):
        for spans in [[], [dict(path="a.py", start=0, end=6)], [dict(path="a.py", start=True, end=2)],
                      [dict(path="a.py", start=0, end=1)] * 2]:
            with self.subTest(spans=spans), self.assertRaises(ValueError): source_positions(self.repo, self.commit, spans)

    def test_payload_cardinality_and_fields(self):
        for payload in [{}, [], {"edits": []}, {"edits": [self.edit()] * 21}, {"edits": "bad"}, {"edits": [self.edit()], "extra": 1}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError): resolve_position_edits(payload, self.prov)

    def test_prompt_schema_budget_and_protocol_isolation(self):
        self.assertIn("000002 | same\n", self.context)
        self.assertNotIn("000001 | outside", self.context)
        self.assertEqual(self.schema["properties"]["edits"]["items"]["properties"]["span_id"]["enum"], self.ids)
        self.assertEqual(generation_limits("source_positions_v1")["Coder"], 4096)
        for role in ("Coder", "Committer"):
            params = sampling_params("source_positions_v1", role, 7, self.schema)
            self.assertEqual(json.loads(params["json_schema"]), self.schema)
            self.assertIn("inclusive", Transcript({}, "", patch_protocol="source_positions_v1").instructions[role])
        self.assertNotIn("json_schema", sampling_params("source_positions_v1", "Planner", 7, self.schema))
        self.assertEqual(generation_limits("unified_diff_v1")["Coder"], 3072)


if __name__ == "__main__":
    unittest.main()
