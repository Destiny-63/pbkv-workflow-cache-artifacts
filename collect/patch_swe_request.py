"""Add metadata-only workflow fields to a pinned stock SGLang request schema.

This does not install PBKV eviction, a selector, or a predictor in the server.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

FIELDS = '''    # PBKV collection metadata only; stock scheduler does not consume these.
    my_chain_id: Optional[str] = None
    my_current_agent: Optional[str] = None
    my_chain_history: Optional[List[str]] = None
    my_is_selector: bool = False
'''


def patched(text):
    matches = list(re.finditer(r"^class GenerateReqInput(?:\([^\n]*\))?:\n", text, re.M))
    if len(matches) != 1 or "my_chain_id:" in text:
        raise ValueError("Expected one unmodified stock GenerateReqInput schema")
    anchor = matches[0].group()
    return text.replace(anchor, anchor + FIELDS, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    original = args.source.read_text()
    result = patched(original)
    compile(result, str(args.out), "exec")
    args.out.write_text(result)
    print(json.dumps({"source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                      "patched_sha256": hashlib.sha256(result.encode()).hexdigest(),
                      "metadata_fields_only": True}))


if __name__ == "__main__":
    main()
