"""Tests for dataset/splits.py: episode-level, deterministic, non-overlapping."""

from dataset.splits import make_splits


def episode_names(n):
    return [f"episode_{i:06d}" for i in range(n)]


def test_every_episode_appears_in_exactly_one_split():
    names = episode_names(50)
    splits = make_splits(names, seed=0)

    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == sorted(names)
    assert len(set(all_assigned)) == len(names)


def test_splits_are_pairwise_disjoint():
    names = episode_names(50)
    splits = make_splits(names, seed=0)

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_split_is_deterministic_given_same_seed():
    names = episode_names(37)
    splits_a = make_splits(names, seed=7)
    splits_b = make_splits(names, seed=7)
    assert splits_a == splits_b


def test_different_seeds_can_produce_different_splits():
    names = episode_names(37)
    splits_a = make_splits(names, seed=1)
    splits_b = make_splits(names, seed=2)
    assert splits_a != splits_b


def test_split_proportions_are_approximately_correct():
    names = episode_names(100)
    splits = make_splits(names, seed=0, train_frac=0.8, val_frac=0.1)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_small_episode_counts_still_partition_completely():
    for n in range(1, 5):
        names = episode_names(n)
        splits = make_splits(names, seed=0)
        assert sorted(splits["train"] + splits["val"] + splits["test"]) == sorted(names)
