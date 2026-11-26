from pbkv_sim.prefix_tree import PrefixTree


def lru_key(n):
    return (n.last_access,)


def test_hit_miss_and_shared_prefix():
    t = PrefixTree(capacity=None)
    a = t.access("w1", "R", [("m", 100), ("c1", 150)], 0.0, on_need=None)
    assert (a.hit_tokens, a.compulsory_tokens) == (0, 150)
    b = t.access("w2", "R", [("m", 100), ("c2", 130)], 1.0, on_need=None)
    assert (b.hit_tokens, b.compulsory_tokens) == (100, 30)
    assert t.used == 180


def test_only_unlocked_leaves_are_candidates_and_parent_follows():
    t = PrefixTree(capacity=None)
    a = t.access("w1", "R", [("m", 100), ("c1", 150)], 0.0, on_need=None)
    t.release(a.path, 1.0)
    victims = t.select(150, lru_key)
    assert [v.key for v in victims] == ["c1", "m"]
    b = t.access("w2", "R", [("m", 100)], 2.0, on_need=None)
    assert [v.key for v in t.select(150, lru_key)] == ["c1"]
    t.release(b.path, 3.0)


def test_refetch_is_attributed_to_eviction_event():
    t = PrefixTree(capacity=None)
    a = t.access("w1", "R", [("m", 100), ("c1", 150)], 0.0, on_need=None)
    t.release(a.path, 1.0)
    t.apply(t.select(50, lru_key), event_id=7)
    again = t.access("w1", "A", [("m", 100), ("c1", 150), ("x", 160)], 2.0, on_need=None)
    assert again.hit_tokens == 100
    assert again.refetch_tokens == 50 and again.refetch_by_event == {7: 50}
    assert again.compulsory_tokens == 10
    assert again.path[1].roles_by_wf["w1"] == {"R", "A"}


def test_capacity_triggers_on_need_with_path_locked():
    t = PrefixTree(capacity=200)
    a = t.access("w1", "R", [("m", 100), ("c1", 150)], 0.0, on_need=None)
    t.release(a.path, 1.0)
    calls = []

    def on_need(need):
        calls.append(need)
        t.apply(t.select(need, lru_key), event_id=1)

    t.access("w2", "R", [("m", 100), ("c2", 190)], 2.0, on_need=on_need)
    assert calls == [40]
    assert t.used == 190
    assert not t.root.children["m"].children["c1"].resident


def test_drop_retired_removes_only_fully_terminated_nodes():
    t = PrefixTree(capacity=None)
    a = t.access("w1", "R", [("m", 100), ("c1", 150)], 0.0, on_need=None)
    b = t.access("w2", "R", [("m", 100)], 0.0, on_need=None)
    t.release(a.path, 1.0)
    t.release(b.path, 1.0)
    t.drop_retired({"w1"})
    assert t.used == 100


def test_peak_locked_and_untagged_reuse_are_tracked():
    t = PrefixTree(capacity=None)
    a = t.access("w1", "R", [("m", 100), ("c1", 150)], 0.0, on_need=None)
    b = t.access("w2", "R", [("m", 100), ("c2", 130)], 0.5, on_need=None)
    assert t.locked == 180 and t.peak_locked == 180
    t.release(a.path, 1.0)
    t.release(b.path, 1.0)
    assert t.locked == 0
    t.access("w1", "A", [("m", 100), ("c1", 150), ("x", 160)], 2.0, on_need=None)
    assert t.reuse_tokens == 150 and t.untagged_reuse_tokens == 150
    assert t.peak_locked == 180
