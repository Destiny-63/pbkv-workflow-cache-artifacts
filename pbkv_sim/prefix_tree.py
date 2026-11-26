"""Radix-style prefix tree at message-segment granularity with leaf-only eviction."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field


@dataclass(eq=False)
class Node:
    id: int
    key: str
    tokens: int
    parent: "Node | None"
    children: dict = field(default_factory=dict)
    resident: bool = False
    ever_resident: bool = False
    lock: int = 0
    last_access: float = 0.0
    roles_by_wf: dict = field(default_factory=dict)
    evicted_by: int | None = None

    def evictable(self) -> bool:
        return self.resident and self.lock == 0 and not any(c.resident for c in self.children.values())

    @property
    def prefix_key(self) -> tuple[str, ...]:
        keys, node = [], self
        while node.parent is not None:
            keys.append(node.key)
            node = node.parent
        return tuple(reversed(keys))


@dataclass
class Access:
    path: list
    hit_tokens: int
    compulsory_tokens: int
    refetch_tokens: int
    refetch_by_event: dict


class PrefixTree:
    def __init__(self, capacity: int | None):
        self.root = Node(-1, "", 0, None, resident=True, ever_resident=True)
        self.capacity = capacity
        self.used = 0
        self.peak = 0
        self.overflow_tokens = 0
        self.locked = 0
        self.peak_locked = 0
        self.reuse_tokens = 0
        self.untagged_reuse_tokens = 0
        self._ids = itertools.count()
        self._resident: set = set()

    def _path(self, segments) -> list:
        node, prev, path = self.root, 0, []
        for key, cum in segments:
            child = node.children.get(key)
            if child is None:
                child = Node(next(self._ids), key, cum - prev, node)
                node.children[key] = child
            path.append(child)
            node, prev = child, cum
        return path

    def _admit(self, nodes, now, on_need) -> tuple[int, int, dict]:
        need = sum(n.tokens for n in nodes)
        if need and self.capacity is not None and self.used + need > self.capacity and on_need is not None:
            on_need(self.used + need - self.capacity)
        compulsory = refetch = 0
        by_event: dict[int, int] = {}
        for n in nodes:
            if n.ever_resident:
                refetch += n.tokens
                by_event[n.evicted_by] = by_event.get(n.evicted_by, 0) + n.tokens
            else:
                compulsory += n.tokens
            n.resident = n.ever_resident = True
            n.evicted_by = None
            self._resident.add(n)
            self.used += n.tokens
        self.peak = max(self.peak, self.used)
        if self.capacity is not None and self.used > self.capacity:
            self.overflow_tokens = max(self.overflow_tokens, self.used - self.capacity)
        return compulsory, refetch, by_event

    def access(self, wf: str, role: str, segments, now: float, on_need) -> Access:
        path = self._path(segments)
        hit = sum(n.tokens for n in path if n.resident)
        missing = [n for n in path if not n.resident]
        for n in path:
            roles = n.roles_by_wf.get(wf)
            if roles is not None:
                self.reuse_tokens += n.tokens
                if role not in roles:
                    self.untagged_reuse_tokens += n.tokens
            if n.lock == 0:
                self.locked += n.tokens
            n.lock += 1
            n.last_access = now
            n.roles_by_wf.setdefault(wf, set()).add(role)
        self.peak_locked = max(self.peak_locked, self.locked)
        compulsory, refetch, by_event = self._admit(missing, now, on_need)
        return Access(path, hit, compulsory, refetch, by_event)

    def release(self, path, now: float) -> None:
        for n in path:
            n.lock -= 1
            if n.lock == 0:
                self.locked -= n.tokens
            n.last_access = now

    def add_output(self, parent, tokens: int, wf: str, role: str, now: float, on_need) -> Node:
        node = Node(next(self._ids), f"out:{next(self._ids)}", tokens, parent)
        parent.children[node.key] = node
        node.last_access = now
        node.roles_by_wf[wf] = {role}
        self._admit([node], now, on_need)
        return node

    def select(self, need: int, key_fn) -> list:
        heap = [(key_fn(n), n.id, n) for n in sorted(self._resident, key=lambda n: n.id) if n.evictable()]
        heapq.heapify(heap)
        gone: set = set()
        victims, freed = [], 0
        while freed < need and heap:
            _, _, n = heapq.heappop(heap)
            if n.id in gone:
                continue
            gone.add(n.id)
            victims.append(n)
            freed += n.tokens
            p = n.parent
            if (p is not self.root and p.resident and p.lock == 0
                    and all((not c.resident) or c.id in gone for c in p.children.values())):
                heapq.heappush(heap, (key_fn(p), p.id, p))
        return victims

    def apply(self, victims, event_id: int) -> None:
        for n in victims:
            if n.resident:
                n.resident = False
                n.evicted_by = event_id
                self._resident.discard(n)
                self.used -= n.tokens

    def drop_retired(self, terminated: set) -> None:
        changed = True
        while changed:
            changed = False
            for n in sorted(self._resident, key=lambda n: n.id):
                if n.evictable() and n.roles_by_wf and set(n.roles_by_wf) <= terminated:
                    self.apply([n], event_id=-1)
                    changed = True
