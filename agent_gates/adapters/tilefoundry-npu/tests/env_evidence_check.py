#!/usr/bin/env python3
"""Reject a verify_environment evidence JSON that must not be cited as ENV evidence.

A run that used a test seam (TFNU_PROBE_OVERRIDE / --negative-control) records
observations but is NOT ENV evidence. So is anything not a clean PASS. The ENV gate
validator must ask THIS question, not "is there an evidence file".
"""
from __future__ import annotations

import json
import sys


def check(path: str) -> list:
    try:
        evidence = json.loads(open(path, encoding="utf-8").read())
    except (OSError, ValueError) as exc:
        return ["UNREADABLE_EVIDENCE: %s" % exc]
    problems = []
    if evidence.get("test_seam_used"):
        problems.append("TEST_SEAM_USED")
    if evidence.get("verdict") != "PASS":
        problems.append("VERDICT_%s" % evidence.get("verdict"))
    if evidence.get("code"):
        problems.append("CODE_%s" % evidence.get("code"))
    if evidence.get("unreaped_children"):
        problems.append("UNREAPED_CHILDREN")
    return problems


def main(argv=None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        print("usage: env_evidence_check.py EVIDENCE.json [...]", file=sys.stderr)
        return 2
    failed = False
    for path in paths:
        problems = check(path)
        if problems:
            failed = True
            print("  FAIL  %s: %s" % (path, ", ".join(problems)))
        else:
            print("  PASS  %s is clean ENV evidence" % path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
