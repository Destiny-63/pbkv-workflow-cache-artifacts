# HoVer-MA call traces

The two files contain paired default and high-difficulty runs for the same 1,500 pseudonymized workflows. Each workflow includes call records and a terminal workflow record.

Each call preserves the role, round and call index, token counts, cache-hit count, prefix segment hashes and timing relationships. Timestamps are seconds relative to the first observed event in that workflow. The files omit claim text, generated prompts and responses, hidden-state paths, raw workflow identifiers and wall-clock timestamps. The split mapping in `../../results/split.json` uses the same pseudonymous IDs.

The call trace files are CC BY 4.0. The source HoVer dataset is MIT licensed and is not redistributed here. See `../LICENSE` and the upstream dataset card for attribution.
