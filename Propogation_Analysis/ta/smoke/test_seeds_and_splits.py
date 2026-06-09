from __future__ import annotations

import unittest

from infra.schema import Problem, SplitAssignment
from infra.seeds import derive_seed
from infra.splits import assert_no_leak, grouped_kfold_assignments, split_counts


class SeedsAndSplitsSmokeTest(unittest.TestCase):
    def test_seed_is_deterministic_and_bounded(self) -> None:
        first = derive_seed(17, "trace", "problem-1")
        second = derive_seed(17, "trace", "problem-1")
        third = derive_seed(17, "trace", "problem-2")
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)

    def test_grouped_split_uses_problem_as_unit(self) -> None:
        problems = [
            Problem(problem_id=f"p{i}", dataset="babi", phase=0, prompt="q", gold_answer="a")
            for i in range(6)
        ]
        assignments = grouped_kfold_assignments(
            problems=problems,
            num_folds=3,
            seed=123,
            held_out_fold=1,
        )
        assert_no_leak(assignments)
        counts = split_counts(assignments)
        self.assertEqual(counts["train"] + counts["test"], 6)
        self.assertGreater(counts["test"], 0)

    def test_leak_assertion_fails_on_role_collision(self) -> None:
        assignments = [
            SplitAssignment(problem_id="p1", fold=0, role="train"),
            SplitAssignment(problem_id="p1", fold=0, role="test"),
        ]
        with self.assertRaises(AssertionError):
            assert_no_leak(assignments)
