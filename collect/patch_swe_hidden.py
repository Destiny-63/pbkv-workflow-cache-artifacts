"""Version-checked, last-prefill-only hidden capture for stock SGLang 0.5.9."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def patch_schedule(text):
    old = "CaptureHiddenMode.FULL\n                if self.return_hidden_states"
    if text.count(old) != 1:
        raise ValueError("Unexpected SGLang capture mode implementation")
    return text.replace(old, old.replace(".FULL", ".LAST"), 1)


def patch_output(text):
    pattern = r"(?P<indent> +)req\.hidden_states\.append\(\s*logits_output\.hidden_states\[\s*hidden_state_offset\s*:\s*\(\s*hidden_state_offset\s*:=\s*hidden_state_offset\s*\+\s*len\(req\.origin_input_ids\)\s*\)\s*\]\s*\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\)\s*\)"
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError("Unexpected SGLang prefill hidden-state slice")
    text = re.sub(pattern, lambda m: m['indent'] + "req.hidden_states.append(\n" + m['indent'] + "    logits_output.hidden_states[i].detach().float().cpu().tolist()\n" + m['indent'] + ")", text, count=1)
    old = "if req.return_hidden_states and logits_output.hidden_states is not None:"
    if text.count(old) != 1:
        raise ValueError("Unexpected SGLang decode hidden-state capture")
    return text.replace(old, "if req.return_hidden_states and logits_output.hidden_states is not None and not req.hidden_states:", 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--originals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {}
    for name, patch in [("schedule_batch.py", patch_schedule), ("scheduler_output_processor_mixin.py", patch_output)]:
        destination = args.out / name
        if destination.exists():
            raise FileExistsError(destination)
        original = (args.originals / name).read_text()
        result = patch(original)
        compile(result, str(destination), "exec")
        destination.write_text(result)
        report[name] = {"original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                        "patched_sha256": hashlib.sha256(result.encode()).hexdigest()}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
