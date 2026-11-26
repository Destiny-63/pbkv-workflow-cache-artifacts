"""Synthetic line-list edits with real Git checks; no benchmark answers."""
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from collect.swe_core import Transcript
from collect.swe_ma import generation_limits, sampling_params
from collect.swe_source_positions import compile_positions, source_positions


class PositionLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / 'fixture.py').write_bytes(b'outside\nsame\nkeep\nsame\nlast\n')
        (self.repo / 'crlf.py').write_bytes(b'outside\r\nsame\r\nlast\r\n')
        self.git('init', '-q'); self.git('add', '.')
        self.git('-c', 'user.name=Synthetic', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-qm', 'isolated synthetic fixture')
        self.commit = self.git('rev-parse', 'HEAD').strip()
        self.displayed = [dict(path='fixture.py', start=1, end=4)]
        self.context, self.schema, self.prov = source_positions(
            self.repo, self.commit, self.displayed, 'source_positions_v2')
        self.sid = next(iter(self.prov['spans']))

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True,
                                       stderr=subprocess.DEVNULL)

    def edit(self, first=2, last=2, lines=None):
        return dict(span_id=self.sid, start_line=first, end_line=last,
                    replace_lines=['changed'] if lines is None else lines)

    def compile(self, edits, prov=None):
        return compile_positions(json.dumps({'edits': edits}), self.repo, self.commit, prov or self.prov)

    def apply(self, diff):
        subprocess.run(['git', '-C', str(self.repo), 'apply', '--check', '-'], input=diff.encode(),
                       capture_output=True, check=True)
        subprocess.run(['git', '-C', str(self.repo), 'apply', '-'], input=diff.encode(),
                       capture_output=True, check=True)
        return (self.repo / 'fixture.py').read_bytes()

    def test_serialization_preserves_all_line_content_and_outside_bytes(self):
        lines = ['    quoted = "中文\\n"', '', '\tsecond']
        diff, resolved = self.compile([self.edit(lines=lines)])
        self.assertEqual(resolved['edits'][0]['replace'], '\n'.join(lines) + '\n')
        self.assertEqual(self.apply(diff), ('outside\n' + '\n'.join(lines) + '\nkeep\nsame\nlast\n').encode())

    def test_empty_array_deletes_and_empty_item_is_a_blank_line(self):
        diff, resolved = self.compile([self.edit(lines=[]), self.edit(4, 4, [''])])
        self.assertEqual(resolved['edits'][0]['replace'], '')
        self.assertEqual(resolved['edits'][1]['replace'], '\n')
        self.assertEqual(self.apply(diff), b'outside\nkeep\n\nlast\n')

    def test_crlf_serialization_and_outside_bytes(self):
        _, _, prov = source_positions(self.repo, self.commit, [dict(path='crlf.py', start=1, end=2)], 'source_positions_v2')
        edit = self.edit(lines=['changed', 'extra']); edit['span_id'] = next(iter(prov['spans']))
        diff, resolved = self.compile([edit], prov)
        self.assertEqual(resolved['edits'][0]['replace'], 'changed\r\nextra\r\n')
        self.apply(diff)
        self.assertEqual((self.repo / 'crlf.py').read_bytes(), b'outside\r\nchanged\r\nextra\r\nlast\r\n')

    def test_internal_terminators_types_and_old_string_field_rejected(self):
        for lines in [['bad\n'], ['bad\r'], ['a\nb'], [1], [None], 'text', True]:
            with self.subTest(lines=lines), self.assertRaises(ValueError): self.compile([self.edit(lines=lines)])
        edit = self.edit(); edit['replace'] = edit.pop('replace_lines')
        with self.assertRaises(ValueError): self.compile([edit])

    def test_noop_and_mixed_noop_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError, 'no effect'): self.compile([self.edit(lines=['same'])])
        with self.assertRaisesRegex(ValueError, 'no effect'):
            self.compile([self.edit(lines=['same']), self.edit(4, 4, ['changed'])])

    def test_bounds_overlap_and_no_implicit_id_or_line_correction(self):
        for changes in [dict(span_id=self.sid+'x'), dict(start_line=1), dict(end_line=5),
                        dict(start_line=4, end_line=2), dict(start_line=True)]:
            edit = self.edit(); edit.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.compile([edit])
        with self.assertRaises(ValueError): self.compile([self.edit(), self.edit()])

    def test_v1_schema_context_and_compiler_remain_separate(self):
        context, schema, prov = source_positions(self.repo, self.commit, self.displayed)
        self.assertEqual(context, self.context)
        self.assertIn('replace', schema['properties']['edits']['items']['properties'])
        self.assertNotIn('replace_lines', schema['properties']['edits']['items']['properties'])
        with self.assertRaises(ValueError): self.compile([self.edit()], prov)
        self.assertEqual(prov['version'], 'visible_source_positions_v1')

    def test_actual_request_schema_role_prompts_and_budgets(self):
        self.assertEqual(self.prov['version'], 'visible_source_positions_v2')
        self.assertEqual(generation_limits('source_positions_v2')['Coder'], 4096)
        for role in ('Coder', 'Committer'):
            request = sampling_params('source_positions_v2', role, 7, self.schema)
            self.assertEqual(json.loads(request['json_schema']), self.schema)
            self.assertIn('replace_lines', Transcript({}, '', patch_protocol='source_positions_v2').instructions[role])
        self.assertNotIn('json_schema', sampling_params('source_positions_v2', 'Planner', 7, self.schema))

    def test_rejected_calls_stay_failed_while_observation_checks_continue(self):
        import numpy as np
        from collect.swe_core import ROLES, digest
        from collect.validate_swe import validate
        data = self.repo / 'isolated_synthetic_trace'
        (data / 'calls').mkdir(parents=True); (data / 'hidden').mkdir()
        (data / 'run_manifest.json').write_text(json.dumps(
            {'purpose': 'isolated synthetic unit-test data', 'patch_protocol': 'source_positions_v2', 'hidden_dimension': 3}))
        (data / 'dataset.json').write_text(json.dumps({'roles': [*ROLES, 'END']}))
        (data / 'synthetic_source_schema.json').write_text(json.dumps({'schema': self.schema, 'provenance': self.prov}))
        (data / 'synthetic_context.txt').write_bytes(self.context.encode())
        rows, history, ids = [], [], [1, 2, 3]
        for i, role in enumerate(('Planner', 'Coder', 'Committer'), 1):
            prefix = data / 'calls' / f'synthetic_{i:02d}'
            request = {'input_ids': ids, 'sampling_params': sampling_params('source_positions_v2', role, 0, self.schema)}
            prefix.with_suffix('.request.json').write_text(json.dumps(request))
            hidden = data / 'hidden' / f'synthetic_{i:02d}.npy'
            np.save(hidden, np.ones(3, dtype=np.float16))
            rows.append(dict(event='call', workflow_id='synthetic', setting='synthetic', role=role,
                             call_index=i, round=1, queue_start=i, prefill_end=i+.1, decode_end=i+.5,
                             prompt_tokens=3, segments=[[digest(struct.pack('<3I', *ids)), 3]],
                             my_chain_history=history.copy(), hidden_file=str(hidden),
                             text=json.dumps({'edits': [self.edit(lines=['same'])]})))
            history.append(role)
        rows.append(dict(event='workflow_end', workflow_id='synthetic', setting='synthetic',
                         instance_id='synthetic_fixture', decode_end=4, verdict='PATCH_APPLY_FAILED'))
        (data / 'swe.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report = validate(data)
        self.assertFalse(report['smoke_acceptance_passed'])
        self.assertFalse(report['source_position_schema_and_ranges_passed'])
        self.assertEqual(report['source_position_editing_calls_verified'], 0)
        self.assertEqual(len(report['source_position_rejected_editing_calls']), 2)
        self.assertTrue(report['history_and_hidden_checks_passed'])
        np.save(hidden, np.array([float('nan'), 1, 1]))
        with self.assertRaisesRegex(ValueError, 'hidden state'): validate(data)


if __name__ == '__main__':
    unittest.main()
