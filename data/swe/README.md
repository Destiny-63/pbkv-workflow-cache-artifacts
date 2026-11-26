# SWE-derived training traces

This directory contains a sanitized subset of model-workflow call traces collected for the SWE-Bench Verified workload.

## Coverage

The release combines four complete trace chunks whose SHA-256 values matched accepted entries in the local available-corpus manifest used for predictor development. It contains 240 workflows from 80 training instances. Each instance has three workflow seeds. The records comprise 2,468 model calls, 2,272 tool events, and 240 workflow-end events.

The underlying corpus used for development was incomplete. The matching local traces cover only part of its training partition. This release has no SWE validation or test traces. It cannot recreate the full predictor-training or cache-replay inputs.

## Fields and redaction

The JSONL records keep the event type, opaque workflow and instance identifiers, train partition, workflow seed, role and call order, model token counts, opaque prefix-segment IDs and token lengths, thinking flag, and relative call/tool timings.

The export removes repository names, issue identifiers, original workflow IDs, prompt and response text, tool feedback and test output, hidden-state paths and vectors, model/server metadata, and absolute timestamps. Fresh keyed pseudonyms are generated for identifiers and prefix-segment hashes. The pseudonym key is not distributed. Timestamps are offset from each workflow's first event.

The split file, `available_train_split.json`, maps the published opaque instance IDs to the train partition. All records in this subset are training traces.

## Read the traces

The files use the generic event schema read by `pbkv_sim.trace.load_workflows`:

```python
from pathlib import Path

from pbkv_sim.trace import load_workflows

workflows = load_workflows(Path("data/swe/available_train.jsonl"))
print(len(workflows))
print(sum(len(workflow.calls) for workflow in workflows.values()))
```

This supports trace parsing and role/timing analysis. Predictor training and the published SWE replay results require additional source traces, hidden vectors, prediction banks, and evaluation artifacts that are not included.

## Source and license

The source benchmark is SWE-Bench Verified. This directory does not redistribute benchmark issue text, repository source code, generated conversations, or test output. The derived metadata trace is shared under CC BY 4.0. The [SWE-bench project](https://github.com/SWE-bench/SWE-bench) and its terms apply to the source benchmark.
