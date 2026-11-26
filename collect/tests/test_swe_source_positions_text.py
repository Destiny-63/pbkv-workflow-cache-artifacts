"""Natural multiline text with explicit boundary serialization and real Git."""
import copy
import json
import subprocess
import unittest

from collect.swe_core import Transcript, digest
from collect.swe_ma import generation_limits, sampling_params
from collect.swe_source_positions import compile_positions, registry_digest, source_positions
import test_swe_source_positions_v2 as fixtures


PROTOCOL = 'source_positions_text_v1'


class MultilineTextTests(unittest.TestCase):
    git = fixtures.PositionLineTests.git
    apply = fixtures.PositionLineTests.apply

    def setUp(self):
        fixtures.PositionLineTests.setUp(self)
        self.context, self.schema, self.prov = source_positions(
            self.repo, self.commit, self.displayed, PROTOCOL)
        self.sid = next(iter(self.prov['spans']))

    def edit(self, replace='changed', first=2, last=2):
        return dict(span_id=self.sid, start_line=first, end_line=last, replace=replace)

    def compile(self, edits, prov=None):
        return compile_positions(json.dumps({'edits': edits}), self.repo, self.commit, prov or self.prov)

    def test_multiline_with_or_without_terminal_newline_has_same_diff(self):
        raw = '    changed = "中文"\n\n\treturn changed'
        first, resolved = self.compile([self.edit(raw)])
        self.assertEqual(first, self.compile([self.edit(raw+'\n')])[0])
        self.assertEqual(resolved['edits'][0]['replace'], raw+'\n')
        self.assertEqual(self.apply(first), ('outside\n'+raw+'\nkeep\nsame\nlast\n').encode())

    def test_raw_and_serialized_evidence_and_extra_blank_lines_retained(self):
        for raw in ['a\nb', 'a\nb\n', 'a\nb\n\n', 'quote"\\n\t\n中文']:
            with self.subTest(raw=raw):
                _, resolved = self.compile([self.edit(raw)])
                edit = resolved['edits'][0]
                ending = '' if raw.endswith('\n') else '\n'
                self.assertEqual(edit['raw_replace'], raw)
                self.assertEqual(edit['raw_replace_sha256'], digest(raw))
                self.assertEqual(edit['replace'], raw+ending)
                self.assertEqual(edit['serialized_replace_sha256'], digest(raw+ending))
                self.assertEqual(edit['boundary_terminator'], ending)
                self.assertEqual(edit['boundary_terminator_added'], bool(ending))

    def test_empty_string_deletes_and_newline_is_one_blank_line(self):
        diff, resolved = self.compile([self.edit(''), self.edit('\n', 4, 4)])
        self.assertFalse(resolved['edits'][0]['boundary_terminator_added'])
        self.assertEqual(self.apply(diff), b'outside\nkeep\n\nlast\n')

    def test_noop_checked_after_serialization_and_not_silently_removed(self):
        for raw in ['same', 'same\n']:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'no effect'):
                self.compile([self.edit(raw)])
        with self.assertRaisesRegex(ValueError, 'no effect'):
            self.compile([self.edit('same'), self.edit('changed', 4, 4)])

    def test_crlf_boundary_preserves_internal_and_outside_bytes(self):
        _, _, prov = source_positions(self.repo, self.commit,
            [dict(path='crlf.py', start=1, end=2)], PROTOCOL)
        raw = '    one\n\ttwo\r\nthree'
        edit = self.edit(raw); edit['span_id'] = next(iter(prov['spans']))
        diff, resolved = self.compile([edit], prov)
        self.assertEqual(resolved['edits'][0]['replace'], raw+'\r\n')
        self.assertEqual(resolved['edits'][0]['boundary_terminator'], '\r\n')
        self.apply(diff)
        self.assertEqual((self.repo/'crlf.py').read_bytes(), ('outside\r\n'+raw+'\r\nlast\r\n').encode())

    def test_supplied_lf_is_not_normalized_to_crlf(self):
        _, _, prov = source_positions(self.repo, self.commit,
            [dict(path='crlf.py', start=1, end=2)], PROTOCOL)
        edit = self.edit('one\ntwo\n'); edit['span_id'] = next(iter(prov['spans']))
        diff, resolved = self.compile([edit], prov)
        self.assertFalse(resolved['edits'][0]['boundary_terminator_added'])
        self.apply(diff)
        self.assertEqual((self.repo/'crlf.py').read_bytes(), b'outside\r\none\ntwo\nlast\r\n')

    def test_original_positions_applied_together_without_offset_inference(self):
        a, b = self.edit('long\naddition'), self.edit('end', 4, 4)
        diff, _ = self.compile([a, b])
        self.assertEqual(diff, self.compile([b, a])[0])
        self.assertEqual(self.apply(diff), b'outside\nlong\naddition\nkeep\nend\nlast\n')

    def test_positions_types_overlap_and_wrong_fields_remain_rejected(self):
        for change in [dict(span_id=self.sid+'x'), dict(start_line=1), dict(end_line=5),
                       dict(start_line=4, end_line=2), dict(start_line=True),
                       dict(replace=['changed']), dict(replace=None), dict(extra='x')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compile([self.edit() | change])
        with self.assertRaises(ValueError): self.compile([self.edit(), self.edit()])
        invalid = self.edit(); invalid['replace_lines'] = invalid.pop('replace')
        with self.assertRaises(ValueError): self.compile([invalid])

    def test_base_visible_hash_and_tracked_mode_checks_not_weakened(self):
        for changed in ('base', 'visible'):
            prov = copy.deepcopy(self.prov)
            if changed == 'base': prov['base_file_sha256']['fixture.py'] = 'wrong'
            else: prov['spans'][self.sid]['source_sha256'] = 'wrong'
            prov['registry_sha256'] = registry_digest(prov)
            with self.subTest(changed=changed), self.assertRaises(ValueError): self.compile([self.edit()], prov)
        (self.repo/'link.py').symlink_to('fixture.py')
        self.git('add', 'link.py')
        self.git('-c', 'user.name=Synthetic', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'synthetic link')
        with self.assertRaises(ValueError):
            source_positions(self.repo, self.git('rev-parse', 'HEAD').strip(), [dict(path='link.py', start=0, end=1)], PROTOCOL)

    def test_old_protocol_rejections_unchanged_and_truncated_json_not_salvaged(self):
        _, _, old = source_positions(self.repo, self.commit, self.displayed)
        with self.assertRaisesRegex(ValueError, 'complete final lines'): self.compile([self.edit()], old)
        with self.assertRaises(json.JSONDecodeError):
            compile_positions('{"edits":[{"replace":"unfinished', self.repo, self.commit, self.prov)

    def test_actual_request_schema_context_and_budgets(self):
        context, schema, _ = source_positions(self.repo, self.commit, self.displayed)
        self.assertEqual(context, self.context)
        self.assertEqual(schema, self.schema)
        self.assertEqual(generation_limits(PROTOCOL)['Coder'], 4096)
        self.assertEqual(generation_limits(PROTOCOL)['Committer'], 4096)
        for role in ('Coder', 'Committer'):
            params = sampling_params(PROTOCOL, role, 7, self.schema)
            self.assertEqual(json.loads(params['json_schema']), self.schema)
            prompt = Transcript({}, '', patch_protocol=PROTOCOL).instructions[role]
            self.assertIn('multiline', prompt)
            self.assertNotIn('replace_lines', prompt)
        self.assertNotIn('json_schema', sampling_params(PROTOCOL, 'Planner', 7, self.schema))


if __name__ == '__main__':
    unittest.main()
