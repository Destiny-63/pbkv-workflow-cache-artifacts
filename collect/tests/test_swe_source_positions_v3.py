"""Explicit optional line-terminator serialization, using synthetic source."""
import unittest

from collect.swe_source_positions import source_positions
import test_swe_source_positions_v2 as fixtures


class OptionalTerminatorTests(unittest.TestCase):
    git = fixtures.PositionLineTests.git
    edit = fixtures.PositionLineTests.edit
    compile = fixtures.PositionLineTests.compile
    apply = fixtures.PositionLineTests.apply

    def setUp(self):
        fixtures.PositionLineTests.setUp(self)
        self.context, self.schema, self.prov = source_positions(
            self.repo, self.commit, self.displayed, 'source_positions_v3')
        self.sid = next(iter(self.prov['spans']))

    def test_optional_terminal_lf_is_not_duplicated(self):
        for lines in [['changed'], ['changed\n']]:
            diff, edits = self.compile([self.edit(lines=lines)])
            self.assertEqual(edits['edits'][0]['replace'], 'changed\n')

    def test_optional_crlf_and_mixed_supplied_bytes_preserved(self):
        _, edits = self.compile([self.edit(lines=['changed\r\n', 'more', '\n'])])
        self.assertEqual(edits['edits'][0]['replace'], 'changed\r\nmore\n\n')

    def test_noop_with_optional_terminator_still_rejected(self):
        for lines in [['same'], ['same\n']]:
            with self.subTest(lines=lines), self.assertRaisesRegex(ValueError, 'no effect'):
                self.compile([self.edit(lines=lines)])

    def test_interior_newlines_multiple_terminators_and_lone_cr_rejected(self):
        for lines in [['bad\nnext'], ['bad\rnext'], ['bad\n\n'], ['bad\r'], ['bad\r\n\r\n']]:
            with self.subTest(lines=lines), self.assertRaises(ValueError): self.compile([self.edit(lines=lines)])

    def test_old_v2_still_rejects_terminal_newline(self):
        _, _, v2 = source_positions(self.repo, self.commit, self.displayed, 'source_positions_v2')
        with self.assertRaises(ValueError): self.compile([self.edit(lines=['changed\n'])], v2)

    def test_real_git_apply_has_no_extra_blank_line(self):
        diff, _ = self.compile([self.edit(lines=['changed\n'])])
        self.assertEqual(self.apply(diff), b'outside\nchanged\nkeep\nsame\nlast\n')
