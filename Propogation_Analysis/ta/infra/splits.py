from __future__ import annotations

import random
from collections import Counter

from .schema import Problem, SplitAssignment


def grouped_kfold_assignments(
    problems: list[Problem],
    num_folds: int,
    seed: int,
    held_out_fold: int,
) -> list[SplitAssignment]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    if not 0 <= held_out_fold < num_folds:
        raise ValueError("held_out_fold must be a valid fold index")

    problem_ids = [problem.problem_id for problem in problems]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("problem_ids must be unique at split time")

    shuffled = list(problem_ids)
    random.Random(seed).shuffle(shuffled)
    assignments: list[SplitAssignment] = []
    for index, problem_id in enumerate(shuffled):
        fold = index % num_folds
        role = "test" if fold == held_out_fold else "train"
        assignments.append(SplitAssignment(problem_id=problem_id, fold=fold, role=role))
    return assignments


def assert_no_leak(assignments: list[SplitAssignment]) -> None:
    role_by_problem: dict[str, str] = {}
    fold_by_problem: dict[str, int] = {}

    for assignment in assignments:
        if assignment.role not in {"train", "test"}:
            raise ValueError(f"invalid split role: {assignment.role}")
        previous_role = role_by_problem.get(assignment.problem_id)
        previous_fold = fold_by_problem.get(assignment.problem_id)
        if previous_role is not None and previous_role != assignment.role:
            raise AssertionError(
                f"problem_id {assignment.problem_id} appears in multiple roles"
            )
        if previous_fold is not None and previous_fold != assignment.fold:
            raise AssertionError(
                f"problem_id {assignment.problem_id} appears in multiple folds"
            )
        role_by_problem[assignment.problem_id] = assignment.role
        fold_by_problem[assignment.problem_id] = assignment.fold


def split_counts(assignments: list[SplitAssignment]) -> Counter[str]:
    return Counter(assignment.role for assignment in assignments)
