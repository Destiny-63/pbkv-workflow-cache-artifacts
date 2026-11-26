"""Versioned, issue-only source retrieval. Never reads reference solutions/tests."""
import ast
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import PurePosixPath

try:
    from .swe_core import git
except ImportError:
    from swe_core import git


STOP = set("the a an to of in on for from is are was were be been being it its this that these those with without and or not but if else as at by can could should would will may might must has have had do does did when then than also any all some no yes true false none self cls def class import return print pass test tests python code example expected actual error issue use using used file line function method object value new old only same see need get set add result works work called call looks like more following below above which what where how why there here we you i he she they them us our your my so into out up down bug fix reproduce version output input want now just let please thanks change behavior instead after before during".split())


def issue_terms(problem):
    # Titles and the opening problem description carry more signal than long
    # environment/package listings or tracebacks later in an issue.
    lead = problem[:2000]
    lead = re.split(r"(?im)^#{1,6}\s*(?:versions|environment|dependencies)\b", lead)[0]
    def words(text):
        return [w for w in re.findall(r"\b[A-Za-z_][A-Za-z_0-9]{2,}\b", text)
                if w.lower() not in STOP and w.strip("_")]
    counts = Counter(words(problem[:8000]))
    counts.update({w: 8 * n for w, n in Counter(words(lead)).items()})
    counts.update({w: 15 * n for w, n in Counter(words(problem.splitlines()[0] if problem else "")).items()})
    return sorted(counts, key=lambda w: (-counts[w], -int("_" in w or any(c.isupper() for c in w)), -len(w), w))[:64]


def is_test_path(path):
    parts = PurePosixPath(path).parts
    return any(p in {"tests", "test", "testing", "example_scripts", "examples"} for p in parts) or PurePosixPath(path).name.startswith("test_")


def source_spans(source, terms, budget):
    """Return complete-line spans; prefer implementation over long class docstrings."""
    lines = source.splitlines(keepends=True)
    if len(source) <= budget:
        return [(0, len(lines))]
    wordsets = [set(re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", line)) for line in lines]
    candidates = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            end = node.end_lineno
            body_words = set().union(*wordsets[start:end])
            score = sum(1 for t in terms if t in body_words)
            score += 20 * int(node.name in terms)
            score += sum(3 for t in terms if t in node.name)
            if isinstance(node, ast.ClassDef):
                if node.name in terms:
                    candidates.append((50, node.lineno - 1, node.lineno))
                # A class can contain a large prose docstring. Rank each real
                # method independently, with the relevant class as context.
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cstart = min([child.lineno] + [d.lineno for d in child.decorator_list]) - 1
                        cwords = set().union(*wordsets[cstart:child.end_lineno])
                        cscore = sum(1 for t in terms if t in cwords)
                        cscore += 20 * int(child.name in terms)
                        cscore += 12 * int(node.name in terms)
                        cscore += 5 * int(child.name == "__init__" and node.name in terms)
                        candidates.append((cscore, cstart, child.end_lineno))
            candidates.append((score, start, end))
    except (SyntaxError, ValueError):
        # Older Python source still receives real snippets, never invented AST.
        pass
    for i, words in enumerate(wordsets):
        score = sum(1 for t in terms if t in words)
        if score:
            candidates.append((score, max(0, i - 8), min(len(lines), i + 100)))
    candidates.sort(key=lambda x: (-x[0], x[2] - x[1], x[1]))
    chosen, used, additions = set(), 0, 0
    for score, start, end in candidates:
        if not score:
            continue
        fresh = [i for i in range(start, end) if i not in chosen]
        if not fresh:
            continue
        cost = sum(len(lines[i]) for i in fresh)
        if cost > budget - used:
            # Do not clip a big definition before its executable body; try a
            # smaller method or a later hit-centered snippet instead.
            continue
        chosen.update(fresh)
        used += cost
        additions += 1
        if additions == 8:
            break
    if not chosen:
        for i, line in enumerate(lines):
            if used + len(line) > budget:
                break
            chosen.add(i); used += len(line)
    spans = []
    for i in sorted(chosen):
        if spans and spans[-1][1] == i:
            spans[-1] = (spans[-1][0], i + 1)
        else:
            spans.append((i, i + 1))
    return spans


def repository_context_v2(repo, commit, problem, max_chars=48000, with_spans=False):
    """Retrieve six source files with rarity-weighted terms and bounded spans."""
    paths = [p for p in git(repo, "ls-tree", "-r", "--name-only", commit).splitlines() if p.endswith(".py")]
    pathset = set(paths)
    terms = issue_terms(problem)
    hits = defaultdict(Counter)
    definitions = defaultdict(set)
    class_definitions = defaultdict(set)
    named_terms = defaultdict(set)
    if terms:
        proc = subprocess.run(["git", "-C", str(repo), "grep", "-n", "-w", "-F",
                               *[a for t in terms for a in ("-e", t)], commit, "--", "*.py"],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr)
        for line in proc.stdout.splitlines():
            fields = line.removeprefix(commit + ":").split(":", 2)
            if len(fields) != 3 or fields[0] not in pathset:
                continue
            path, _, content = fields
            words = set(re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", content))
            for term in terms:
                if term in words:
                    hits[path][term] += 1
            match = re.search(r"\b(?:class|def)\s+(\w+)", content)
            if match:
                pieces = set(match[1].strip("_").split("_"))
                named_terms[path].update(t for t in terms if t in pieces)
            if match and match[1] in terms:
                definitions[path].add(match[1])
                if re.search(r"\bclass\s+" + re.escape(match[1]) + r"\b", content):
                    class_definitions[path].add(match[1])
    df = Counter(t for counter in hits.values() for t in counter)
    # Example names like test_foo.py do not trump real implementation symbols.
    explicit = set(re.findall(r"[\w./-]+\.py", problem))
    resolved = {p for p in paths if any(p == e or ("/" in e and p.endswith("/" + e)) for e in explicit)}

    def score(path):
        lexical = sum((1 + math.log1p(min(count, 5))) * math.log1p(len(paths) / df[t]) / (1 + terms.index(t) / 8)
                      for t, count in hits[path].items())
        lexical += sum(100 / (1 + terms.index(t) / 8) for t in definitions[path])
        lexical += sum(1000 / (1 + terms.index(t) / 8) for t in class_definitions[path])
        lexical += sum(40 / (1 + terms.index(t) / 8) for t in named_terms[path])
        lexical += 5 * sum(t.lower() in path.lower() for t in terms)
        lexical += 50 * int(path in resolved)
        return lexical * (0.25 if is_test_path(path) else 1)

    selected = sorted((p for p in paths if hits[p] or p in resolved), key=lambda p: (-score(p), p))[:6]
    parts, total, displayed = [], 0, []
    for rank, path in enumerate(selected):
        source = git(repo, "show", f"{commit}:{path}")
        lines = source.splitlines(keepends=True)
        # Header/omission metadata fits within the same hard overall budget.
        budget = min((14000, 10000, 8000, 6000, 5000, 4000)[rank], max_chars - total - 1000)
        if budget <= 0:
            break
        for start, end in source_spans(source, terms, budget):
            header = f"\n--- {path} (base commit, lines {start + 1}-{end}; literal source, no line labels) ---\n"
            block = header + "".join(lines[start:end])
            if total + len(block) > max_chars:
                break
            parts.append(block); total += len(block)
            displayed.append({"path": path, "start": start, "end": end})
    text = "".join(parts) or "No matching repository source was found."
    return (text, displayed) if with_spans else text
