"""Hierarchy distance on a small made-up hierarchy. Run: python -m pytest test_hierarchy.py -q

        P (Procedure)
        /           \\
      A (Operation on ovary)   X (Operation on uterus)
     /   \\
    B     C          B = Biopsy of ovary, C = Incision of ovary
    |
    D                D = Incisional biopsy of ovary; D also has a second parent C
"""
import pytest

from hierarchy import Hierarchy, level_label

EDGES = {  # descendant: {ancestor: min levels}
    "P": {"P": 0},
    "A": {"A": 0, "P": 1},
    "X": {"X": 0, "P": 1},
    "B": {"B": 0, "A": 1, "P": 2},
    "C": {"C": 0, "A": 1, "P": 2},
    "D": {"D": 0, "B": 1, "C": 1, "A": 2, "P": 3},
}


@pytest.fixture
def h():
    return Hierarchy(EDGES)


def test_exact(h):
    r = h.relation("B", {"B"})
    assert (r["kind"], r["distance"], level_label(r)) == ("exact", 0, "R0")


def test_child_is_more_specific_one_step(h):
    r = h.relation("D", {"B"})
    assert (r["kind"], r["distance"], level_label(r)) == ("more_specific", 1, "R1")


def test_parent_is_more_general_one_step(h):
    r = h.relation("A", {"B"})
    assert (r["kind"], r["distance"], level_label(r)) == ("more_general", 1, "R1")


def test_sibling_is_lateral_two_steps(h):
    r = h.relation("C", {"B"})
    assert (r["kind"], r["distance"], r["up_pred"], r["up_gold"]) == ("lateral", 2, 1, 1)


def test_grandparent_two_steps(h):
    r = h.relation("A", {"D"})
    assert (r["kind"], r["distance"]) == ("more_general", 2)


def test_second_parent_gives_shortest_path(h):
    # D is a child of C as well, so C -> D is one step even though the path via B would be longer
    r = h.relation("C", {"D"})
    assert (r["kind"], r["distance"]) == ("more_general", 1)


def test_distant_concept_is_further(h):
    r = h.relation("X", {"D"})
    assert r["distance"] == 4 and level_label(r) == "further"


def test_closest_of_several_targets_counts(h):
    r = h.relation("A", {"X", "B"})
    assert (r["target"], r["distance"]) == ("B", 1)


def test_missing_and_unknown(h):
    assert level_label(h.relation(None, {"B"})) == "none"
    r = h.relation("UNKNOWN", {"B"})
    assert r["kind"] == "not_in_hierarchy" and level_label(r) == "further"


def test_no_common_ancestor():
    h = Hierarchy({"a": {"a": 0}, "b": {"b": 0}})
    assert h.relation("a", {"b"})["kind"] == "unrelated"
