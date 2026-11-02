#!/usr/bin/env python3
"""Independent acceptance for a BF16 RMSNorm candidate on Ascend950PR.

Scope
-----

This script is the accept Gate's judge. It does not trust anything a run says
about itself: it re-reads the archived candidate source, re-compiles it in a
fresh interpreter against an EMPTY kernel cache, re-measures it against a
high-precision definition it computes itself, and re-measures the vendor
reference in the SAME measurement window. Everything it reports is something it
observed.

Three parties, kept apart on purpose
------------------------------------

* the **math judge** - this file's own float64 definition of the operator. It
  decides correctness and is never a speed opponent.
* the **vendor baseline** - torch_npu.npu_rms_norm(x, w, 1e-6)[0], measured in the
  same window as the candidate. That is the competition.
* the **candidate** - a real TileLang kernel, recompiled from the archive.

Collapsing any two of those is how a run ends up optimising toward a number that
is not the target.

Why the truth is computed on the CPU
------------------------------------

.float64() / .double() on this device can silently yield float32 (a float32
tensor .double()d comes back float32; a float64 ALLOCATION is genuinely float64,
which makes the trap worse - it looks right until it is a cast). A distance
"computed in float64" on the device may therefore be float32-accurate while
claiming headroom it does not have. The truth and every distance in this file are
computed on the CPU in true float64, after copying the outputs back. The copy
cost is excluded from every device timing: timing reads msprof kernel durations,
which never contain host work.

Why timing is paired inside one window
--------------------------------------

The machine is shared. Absolute device time drifts by more than 2x between runs
as co-tenants come and go (measured: the same vendor kernel read 0.0206 ms and
0.0514 ms in two runs), so a ratio between two runs' absolutes is not evidence.
Inside one window the paired ratio is stable to about +/-8 percent, and a
verified co-tenant that kept the card 90 percent busy moved the paired ratio by
only 3.3 percent. So A and B are measured back to back inside one window, the
order alternates, and the verdict is computed from the paired differences.

Every window carries a witness - the card's AICore utilisation around the window
and, when one is supplied, a co-tenant heartbeat. A window whose witness cannot
be read is VOID rather than silently counted: a window that cannot be classified
afterwards is exactly the case that once produced an unexplained +70 percent
shift.

Output contract
---------------

Standard output carries exactly one JSON object and nothing else; diagnostics go
to stderr. The object is bounded at 64 KiB (the evaluator's limit). If it would
exceed that, redundant per-window fields are dropped first and named in
sizeGuard; the raw samples, the intervals and the verdict are never dropped.

Exit codes
----------

  0   MATCH or EXCEED - the candidate met the frozen criterion
  1   FAIL - a real defect (wrong numbers, missing evidence, slower than target)
  2   a configuration or usage error: nothing was judged
  4   RETEST - the evidence was too noisy to decide; not a failure
  75  BLOCKED_ENV - no usable device: nothing was measured
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import threading
import time

PROG = 'accept-rmsnorm'
SCHEMA = 1
PROTOCOL = 'forgekernel-ascend-npu/accept/v1'

# ── frozen acceptance constants ─────────────────────────────────────────────
#
# Declared before any candidate is measured and never adjusted afterwards. A
# threshold moved after seeing a result is not a criterion.

D_MODEL = 4096
EPS = 1e-6
DTYPE = 'bfloat16'
SHAPES = {'8192x4096': (8192, D_MODEL), '64x4096': (64, D_MODEL)}

# Precision, mirrored from ForgeKernel's card eval/precision.py (K_RATIO,
# FLOOR_ULPS, FLOOR_REL) so this Gate and the in-loop judge cannot disagree about
# what "close enough" means. The unit roundoff of bfloat16 is 2**-8 (8 significand
# bits); the task's own 2**-7 is that doubled, and is the largest relative error a
# single rounding of a normal bf16 value can produce. Both are named below.
K_RATIO = 3.0
FLOOR_ULPS = 1.0
FLOOR_REL = 2.0 ** -17
U_BF16 = 2.0 ** -8
TASK_FLOOR = 2.0 ** -7
# A candidate's worst per-element error may be at most this multiple of the
# vendor's worst on the SAME inputs, and never below the dtype floor. This is the
# "same standard on both sides" clause: an absolute bound alone would fail an
# honest kernel whose reduction order merely differs, and a ratio alone would let
# a sloppy vendor mask a sloppy candidate.
MAX_REL_MARGIN = 1.5

# Held-out input families. 'normal' is what a benchmark would sample; the other two
# reach the range shortcuts a natural draw does not. They exist because a
# candidate tuned only to the published shape can still be wrong.
FAMILIES = ('normal', 'near_zero_rows', 'outlier_channels')

# Timing. speedup = vendor_ms / candidate_ms, so 1.0 means the same device time.
MATCH_POINT = 1.0
MATCH_LOWER = 0.98       # 2 percent pre-declared engineering-equivalence tolerance
EXCEED_LOWER = 1.0
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 20261102
# Below this many USABLE windows a shape has no verdict (RETEST), never a FAIL.
MIN_WINDOWS = 9
# AICore utilisation at or above this makes a window 'contended'. Recorded, not
# gated: the ratio is still used, and the verdict carries the contention level.
CONTENDED_UTIL_PCT = 20
# p90/p10 of the usable speedups above this means the windows disagree with each
# other more than the 2 percent tolerance can absorb: RETEST, not FAIL.
MAX_RATIO_SPREAD = 1.25

# A canary kernel is measured inside every window and compared against the run's own
# median. A canary that inflates means the CARD was busy while this window ran, which
# is contention observed at kernel granularity instead of inferred from a slow poll.
#
# This exists because npu-smi cannot be sampled fast: one invocation costs 0.5-3.2 s
# on this host (measured), so a 100-200 ms trajectory is not physically available and
# claiming one would be a fabricated witness. A canary has no sampling-rate problem:
# it measures the thing we actually care about, at the resolution the measurement
# already runs at.
CANARY_INFLATION = 1.15
# Below this many samples per second of window time, a trajectory is too sparse to
# support a 'quiet' label. Reported, not assumed.
MIN_WITNESS_SAMPLES_PER_SECOND = 0.5

# Contended and sparse windows are recorded but EXCLUDED from the decision sample.
#
# This is not caution, it is arithmetic. Measured on this host with an unchanged
# candidate under a real sustained tenant: the paired ratio moved 0.3224 -> 0.6615
# at 8192x4096 (x2.05) and 0.9939 -> 0.8273 at 64x4096 (x0.83). The bias REVERSES
# direction between shapes, because the two arms suffer unequally and which arm
# suffers more depends on the shape. A systematic 2.05x shift of that kind is
# enough to turn a genuinely 2x-slower candidate into a 'MATCH': 0.478 * 2.05 = 0.98.
# A median resists noise; it does not resist bias, so the biased windows must leave
# the sample rather than be averaged into it.
MIN_QUIET_FRACTION = 0.7

# host_bound_ratio = host_enqueue_ms / device_median_ms. At or above this the
# 'device' reading is host dispatch cost, so a device baseline taken from it is
# invalid. Same threshold as ForgeKernel's timing_audit.HOST_BOUND_RATIO_WARN.
HOST_BOUND_RATIO_WARN = 0.5

MAX_OUTPUT_BYTES = 64 * 1024

# ── authenticity policy ─────────────────────────────────────────────────────
#
# A candidate must compute RMSNorm in its own TileLang kernel. 'import torch_npu'
# is explicitly ALLOWED: the backend has to be registered before an npu tensor can
# exist at all, and events / allocation / cast helpers come from there. What is
# refused is delegating the COMPUTATION to a vendor fusion operator.
#
# A name deny-list alone is not enough: an aliased call built with importlib +
# getattr + string concatenation resolves to the same operator at run time and
# matches no name pattern. So this file walks the AST, folds constants, and
# refuses what it cannot resolve.
VENDOR_MODULE_ROOTS = ('torch_npu', 'acl', 'aclnn', 'torchair')
FORBIDDEN_OPS = frozenset({
    'npu_rms_norm', 'npu_add_rms_norm', 'npu_rms_norm_quant', 'npu_rms_norm_cast',
    'npu_layer_norm', 'npu_layernorm', 'npu_layer_norm_backward',
    'npu_layer_norm_quant', 'npu_group_norm', 'npu_rms_norm_backward',
})
FORBIDDEN_PATTERNS = (re.compile(r'rms.*norm'), re.compile(r'layer.*norm'))
FORBIDDEN_LITERALS = ('aclnnRmsNorm', 'aclnnAddRmsNorm', 'aclnnLayerNorm')

# A kernel this toolchain generated carries a name ending in _kernel (tilelang
# names the entry so). The runtime witness requires one.
TILELANG_KERNEL_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*_kernel$')


class AcceptError(Exception):
    """A configuration or usage error: nothing was judged."""

    EXIT = 2


class BlockedEnv(AcceptError):
    """No usable device: nothing was measured."""

    EXIT = 75


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def sha256_tree(root: Path, suffixes=('.py',)) -> dict:
    """Every candidate file under root, by relative path and sha256."""
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        files[str(path.relative_to(root))] = digest_file(path)
    return files


def number(value):
    """A JSON-safe number: inf becomes the string 'inf', NaN becomes null."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'inf' if value > 0 else '-inf'
        return round(value, 9)
    return value



# ═══════════════════════════════════════════════════════════════════════════
# A. authenticity: does the candidate really compute in its own kernel?
# ═══════════════════════════════════════════════════════════════════════════


class Findings:
    """What the reachability walk saw, split by how it bears on the verdict."""

    def __init__(self):
        self.delegations = []      # a vendor fusion op IS reached: fatal
        self.unresolved = []       # a dynamic name that could be anything: fatal
        self.warnings = []         # recorded, not fatal
        self.imports = []          # every module the artifact imports
        self.strings = []          # vendor-op literals that need no call to matter
        self.files = []

    def add(self, kind, file, line, detail):
        getattr(self, kind).append({'file': file, 'line': line, 'detail': detail})


class _Folder:
    """Constant folding over the subset of Python a kernel artifact actually uses.

    Deliberately conservative: anything it cannot evaluate returns None, and a
    None in a dispatch position is a refusal, not a pass.
    """

    def __init__(self):
        self.names = {}
        self.module_aliases = {}

    def value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.JoinedStr):
            out = []
            for piece in node.values:
                if isinstance(piece, ast.Constant):
                    out.append(str(piece.value))
                elif isinstance(piece, ast.FormattedValue):
                    inner = self.value(piece.value)
                    if inner is None:
                        return None
                    out.append(str(inner))
                else:
                    return None
            return ''.join(out)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, int) and isinstance(right, int):
                return left + right
            return None
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.value(node.value)
            if isinstance(base, str):
                return base + '.' + node.attr
            return None
        # '.'.join([...]) - the usual hiding place for a split name
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != 'join' or len(node.args) != 1:
                return None
            separator = self.value(node.func.value)
            items = node.args[0]
            if not isinstance(separator, str) or not isinstance(items, (ast.List, ast.Tuple)):
                return None
            out = []
            for element in items.elts:
                item = self.value(element)
                if not isinstance(item, str):
                    return None
                out.append(item)
            return separator.join(out)
        return None


def _import_module_name(folder, node):
    """The module name that an import call names, when it is a constant one.

        backend = importlib.import_module("torch" + "_npu")

    Binding that result to the alias is what keeps the walk from going blind
    exactly where the aliasing trick lives: without it, every later use of
    'backend' is an unresolvable name.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    is_import = ((isinstance(func, ast.Attribute)
                  and func.attr in ('import_module', 'module_from_spec'))
                 or (isinstance(func, ast.Name) and func.id == '__import__'))
    if not is_import:
        return None
    argument = node.args[0] if node.args else None
    resolved = folder.value(argument) if argument is not None else None
    return resolved if isinstance(resolved, str) else None


def _resolved_owner(folder, node):
    """The dotted module an expression names, folding local aliases once more."""
    resolved = folder.value(node)
    if not isinstance(resolved, str):
        return None
    return folder.module_aliases.get(resolved, resolved)


def _note_assignment(folder, node):
    for target in node.targets:
        if not isinstance(target, ast.Name):
            continue
        imported = _import_module_name(folder, node.value)
        if imported is not None:
            # The name IS that module for every later use of it.
            folder.names[target.id] = imported
            folder.module_aliases[target.id] = imported
            continue
        # getattr(<module>, <constant>) binds to 'module.attr', so a later call
        # through that alias is still attributable to the operator it names.
        value_node = node.value
        if (isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name)
                and value_node.func.id == 'getattr' and len(value_node.args) >= 2):
            owner = _resolved_owner(folder, value_node.args[0])
            attribute = folder.value(value_node.args[1])
            if isinstance(owner, str) and isinstance(attribute, str):
                folder.names[target.id] = owner + '.' + attribute
                continue
        value = folder.value(value_node)
        if value is None:
            continue
        folder.names[target.id] = value if isinstance(value, str) else str(value)


def _vendor_root(module):
    if not module:
        return None
    head = module.split('.')[0]
    return head if head in VENDOR_MODULE_ROOTS else None


def _is_forbidden_name(name):
    if name in FORBIDDEN_OPS:
        return True
    return any(pattern.search(name) for pattern in FORBIDDEN_PATTERNS)


def _in_source_order(tree):
    """Every node, in the order a reader would meet it (line first, then column)."""
    nodes = [node for node in ast.walk(tree)
             if getattr(node, 'lineno', None) is not None]
    nodes.sort(key=lambda node: (node.lineno, node.col_offset))
    return nodes


def _walk_function(node):
    """Every node inside one function body, in source order."""
    nodes = []
    for child in node.body:
        nodes.extend(_in_source_order_under(child))
    nodes.sort(key=lambda item: (item.lineno, item.col_offset))
    return nodes


def _in_source_order_under(node):
    return [item for item in ast.walk(node) if getattr(item, 'lineno', None) is not None]


def scan_source(name, text, findings):
    """Walk one archived module and record what it could compute with."""
    try:
        tree = ast.parse(text, filename=name)
    except SyntaxError as exc:
        findings.add('unresolved', name, exc.lineno or 0,
                     'the module does not parse, so its reachable calls cannot be '
                     'established: ' + str(exc.msg))
        return
    findings.files.append(name)
    folder = _Folder()

    # Bindings are resolved in SOURCE order (imports, then assignment by
    # assignment), because an alias built from an earlier binding is only
    # resolvable once that binding is known. A breadth-first walk would visit a
    # use before its definition and call it unresolvable.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                findings.imports.append(alias.name)
                bound = alias.asname or alias.name.split('.')[0]
                folder.module_aliases[bound] = alias.name
                folder.names[bound] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            findings.imports.append(node.module)
            for alias in node.names:
                full = node.module + '.' + alias.name
                bound = alias.asname or alias.name
                folder.module_aliases[bound] = full
                folder.names[bound] = full
    for node in _in_source_order(tree):
        if isinstance(node, ast.Assign):
            _note_assignment(folder, node)
    for node in _in_source_order(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in _walk_function(node):
                if isinstance(inner, ast.Assign):
                    _note_assignment(folder, inner)

    def module_of(node):
        """The dotted module an expression names, when it is a resolvable one."""
        resolved = folder.value(node)
        if not isinstance(resolved, str):
            return None
        return folder.module_aliases.get(resolved, resolved)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for literal in FORBIDDEN_LITERALS:
                if literal in node.value:
                    findings.add('strings', name, getattr(node, 'lineno', 0),
                                 'the source names the vendor operator ' + literal)
        if isinstance(node, ast.Attribute):
            owner = module_of(node.value)
            if _vendor_root(owner) and _is_forbidden_name(node.attr):
                findings.add('delegations', name, node.lineno,
                             'reaches ' + str(owner) + '.' + node.attr +
                             ' (vendor fusion operator)')
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        dynamic_import = (
            (isinstance(func, ast.Attribute)
             and func.attr in ('import_module', 'module_from_spec'))
            or (isinstance(func, ast.Name) and func.id == '__import__'))
        if dynamic_import:
            argument = node.args[0] if node.args else None
            resolved = folder.value(argument) if argument is not None else None
            if isinstance(resolved, str):
                findings.imports.append(str(resolved))
                if _vendor_root(str(resolved)):
                    findings.add('warnings', name, node.lineno,
                                 'imports the vendor backend dynamically: '
                                 + str(resolved) + ' (allowed for registration, never '
                                 'for computation)')
            else:
                findings.add('unresolved', name, node.lineno,
                             'imports a module whose name is not a constant, so the '
                             'reachable code set cannot be established')
        if isinstance(func, ast.Name) and func.id == 'getattr' and len(node.args) >= 2:
            owner = module_of(node.args[0])
            attribute = folder.value(node.args[1])
            if _vendor_root(owner):
                if isinstance(attribute, str) and _is_forbidden_name(attribute):
                    findings.add('delegations', name, node.lineno,
                                 'resolves ' + str(owner) + '.' + attribute +
                                 ' through getattr (alias of a vendor fusion operator)')
                elif not isinstance(attribute, str):
                    findings.add('unresolved', name, node.lineno,
                                 'resolves an attribute of ' + str(owner) +
                                 ' from a name that is not a constant')
        if isinstance(func, ast.Name) and func.id in ('eval', 'exec'):
            argument = node.args[0] if node.args else None
            if folder.value(argument) is None:
                findings.add('unresolved', name, node.lineno,
                             func.id + '() of a non-constant expression: the executed '
                             'code is not in the archive')
        if isinstance(func, ast.Name):
            resolved = folder.names.get(func.id)
            if isinstance(resolved, str) and _vendor_root(resolved):
                if _is_forbidden_name(resolved.split('.')[-1]):
                    findings.add('delegations', name, node.lineno,
                                 'calls the vendor fusion operator ' + resolved)
            elif not isinstance(resolved, str) and func.id in FORBIDDEN_OPS:
                findings.add('delegations', name, node.lineno,
                             'calls ' + func.id + ' (a vendor fusion operator name)')


def scan_artifact(root):
    """Scan every module the archive ships, and say what it reaches outside.

    The archive is self-contained by contract: a helper it needs must be inside
    it, so every .py file it ships is reachable by definition and is scanned.
    Imports that resolve outside the archive are recorded, because they are the
    only code this walk cannot see.
    """
    root = Path(root)
    findings = Findings()
    files = sorted(path for path in root.rglob('*.py') if path.is_file())
    if not files:
        raise AcceptError('the archived candidate ships no Python source at ' + str(root))
    for path in files:
        scan_source(str(path.relative_to(root)),
                    path.read_text(encoding='utf-8', errors='replace'), findings)
    shipped_stems = {path.stem for path in files}
    external = sorted({name.split('.')[0] for name in findings.imports
                       if name and not name.startswith('_')})
    inside = sorted(name for name in external if name in shipped_stems)
    return findings, {
        'root': str(root),
        'files': sha256_tree(root),
        'externalImports': sorted(name for name in external if name not in shipped_stems),
        'internalImports': inside,
    }


def judge_authenticity(scan, artifact, runtime, compile_evidence):
    """Did the candidate compute RMSNorm in its own TileLang kernel?

    Three independent legs, all required. Either alone is defeatable:

    * static reachability catches a delegation the source spells out, including
      through aliases and constant-folded names;
    * the compile artifact proves a device binary was produced from the archived
      source during THIS acceptance, in a cache that started empty;
    * the runtime witness names the kernels that ran, so a source that never
      launches its kernel cannot pass on a file that merely exists.
    """
    reasons = []
    if scan.delegations:
        reasons.append('the candidate delegates its computation to a vendor fusion '
                       'operator: ' + '; '.join(
                           item['file'] + ':' + str(item['line']) + ' ' + item['detail']
                           for item in scan.delegations[:4]))
    if scan.unresolved:
        reasons.append('the candidate resolves code or names at run time in a way the '
                       'archive cannot bound: ' + '; '.join(
                           item['file'] + ':' + str(item['line']) + ' ' + item['detail']
                           for item in scan.unresolved[:4]))
    if scan.strings:
        reasons.append('the source names a vendor operator literal: ' + '; '.join(
            item['file'] + ':' + str(item['line']) + ' ' + item['detail']
            for item in scan.strings[:4]))
    compile_ok = bool(compile_evidence and compile_evidence.get('aibinCount', 0) >= 1)
    if not compile_ok:
        reasons.append('no new device binary (.aibin) appeared in the empty kernel '
                       'cache while the archived source was compiled, so no kernel of '
                       'the candidate own was produced during acceptance')
    if runtime is not None:
        if runtime.get('vendorOpObserved'):
            reasons.append('the candidate timed run launched the vendor operator: '
                           + ', '.join(runtime.get('vendorOperatorNames') or [])[:200])
        elif not runtime.get('launchedTilelangKernel'):
            reasons.append('the candidate timed run launched no TileLang-generated kernel')
    return {
        'verdict': 'pass' if not reasons else 'fail',
        'reasons': reasons,
        'static': {
            'filesScanned': scan.files,
            'delegations': scan.delegations,
            'unresolvedDynamic': scan.unresolved,
            'warnings': scan.warnings,
            'vendorLiterals': scan.strings,
            'imports': sorted(set(scan.imports)),
        },
        'artifact': {
            'files': artifact.get('files'),
            'externalImports': artifact.get('externalImports'),
            'internalImports': artifact.get('internalImports'),
            'compile': compile_evidence,
        },
        'runtimeWitness': runtime,
    }



# ═══════════════════════════════════════════════════════════════════════════
# B. precision: is the candidate arithmetic the contract arithmetic?
# ═══════════════════════════════════════════════════════════════════════════


def ulp_multiple(value):
    """A max_rel expressed in bf16 ulps of 2**-7 - the unit that carries meaning here.

    2**-7 is exactly one ulp of bf16 at the top of a binade, and a correctly rounded
    bf16 result can land EXACTLY on it. Comparisons against it must therefore use
    <=, never <: a strict comparison would fail the vendor itself and every correct
    implementation that happens to land on the boundary. A correct implementation
    reads 1.0 ulp; a bf16-accumulating one reads 3.4 to 5.5 ulp, and that is the
    separation the gate relies on - not any percentage margin.
    """
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(value / TASK_FLOOR, 4)


def precision_limit(e_ref, dtype_unit=U_BF16):
    """The relative-L2 limit: K_RATIO x the reference own error, plus a floor.

    The floor keeps an exact reference from failing every kernel that rounds once;
    it is never below one unit of the output dtype roundoff.
    """
    base = e_ref if math.isfinite(e_ref) else math.inf
    limit = K_RATIO * base + max(FLOOR_ULPS * dtype_unit, FLOOR_REL)
    return limit if math.isfinite(limit) else math.inf


def judge_precision(rows):
    """Apply the frozen criterion to measured rows, per shape and input family.

    Two rules, both required. Neither is a tolerance on absolute error, which
    saturates at the bf16 floor for honest and sloppy kernels alike:

      P1  e_cand <= K_RATIO * e_ref + floor     (relative L2 against fp64 truth)
      P2  max_rel_cand <= max(2**-7, 1.5 * max_rel_ref)   (same standard both sides)

    e_ref is the vendor own relative L2 error on the same inputs: an honest kernel
    may not be much further from the truth than the competition is.
    """
    judged, failures = [], []
    for row in rows:
        shape, family = row['shape'], row['family']
        cand, ref, truth = row['candidate'], row['vendor'], row['truth']
        key = shape + '/' + family
        if truth != 'float64' or cand is None or ref is None:
            failures.append(key + ': no float64 truth was measured, so no precision '
                                  'claim can be made')
            judged.append({'shape': shape, 'family': family, 'status': 'unavailable',
                           'why': str(truth)})
            continue
        e_ref, e_cand = ref['rel_l2'], cand['rel_l2']
        limit = precision_limit(e_ref)
        ref_max = ref['max_rel']
        max_limit = max(TASK_FLOOR, MAX_REL_MARGIN * (ref_max if math.isfinite(ref_max)
                                                      else 0.0))
        passed_p1 = math.isfinite(e_cand) and e_cand <= limit
        passed_p2 = math.isfinite(cand['max_rel']) and cand['max_rel'] <= max_limit
        entry = {
            'shape': shape, 'family': family,
            'status': 'pass' if (passed_p1 and passed_p2) else 'fail',
            'truth': truth,
            'e_cand': number(e_cand), 'e_ref': number(e_ref), 'limit_l2': number(limit),
            'ratio_to_limit': number(e_cand / limit) if math.isfinite(limit) and limit else None,
            # Absolute, relative AND ulp multiples. The ulp column is the one that
            # carries meaning: 2**-7 is exactly one bf16 ulp at the top of a binade,
            # so a correct implementation reads 1.0 and a bf16-accumulating one
            # reads 3.4-5.5. Reported as x/2**-7.
            'max_rel_cand': number(cand['max_rel']), 'max_rel_ref': number(ref_max),
            'max_rel_limit': number(max_limit),
            'max_abs_cand': number(cand.get('max_abs')),
            'mean_abs_cand': number(cand['mean_abs']), 'mean_abs_ref': number(ref['mean_abs']),
            'ulp_cand': ulp_multiple(cand['max_rel']),
            'ulp_ref': ulp_multiple(ref_max),
            'ulp_limit': ulp_multiple(max_limit),
            'ulp_floor': 1.0,
            'rules': {'relL2': passed_p1, 'maxRel': passed_p2},
        }
        judged.append(entry)
        if entry['status'] == 'fail':
            failures.append(
                key + ': ' + ('relL2 ' + str(entry['e_cand']) + ' > ' + str(entry['limit_l2'])
                              if not passed_p1 else
                              'maxRel ' + str(entry['max_rel_cand']) + ' > '
                              + str(entry['max_rel_limit'])))
    scored = [item for item in judged if item.get('ratio_to_limit') is not None]
    worst = max(scored, key=lambda item: item['ratio_to_limit']) if scored else None
    return {'verdict': 'pass' if judged and not failures else 'fail',
            'failures': failures, 'rows': judged, 'worst': worst,
            'rule': ('P1 e_cand <= ' + str(K_RATIO) + '*e_ref + max(' + str(FLOOR_ULPS)
                     + '*2**-8, 2**-17); P2 max_rel_cand <= max(2**-7, '
                     + str(MAX_REL_MARGIN) + '*max_rel_ref)')}


# ═══════════════════════════════════════════════════════════════════════════
# C. timing: one window, two arms, a witness on every window
# ═══════════════════════════════════════════════════════════════════════════


def bootstrap_median_interval(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """95 percent percentile interval of the median, from the samples themselves."""
    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], values[0])
    generator = random.Random(seed)
    out = []
    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))] for _ in range(len(values))]
        out.append(statistics.median(sample))
    out.sort()
    return (out[int(0.025 * resamples)], out[min(resamples - 1, int(0.975 * resamples))])


def classify_window(witness):
    """Label a window from what was observed around it.

    A window that cannot be classified afterwards is VOID: it is the exact case
    that once produced an unexplained +70 percent shift, and no amount of
    averaging makes an unlabelled window interpretable.
    """
    before = witness.get('aicoreUtilBefore')
    after = witness.get('aicoreUtilAfter')
    heartbeat = witness.get('heartbeatAdvanced')
    co_tenants = witness.get('coTenants')
    trajectory = witness.get('aicoreUtilTrajectory')
    # Two independent witnesses: a utilisation TRAJECTORY sampled through the
    # window (does the card actually get busy while we measure?), and the driver's
    # process table (did somebody else's process appear or vanish?). A window with
    # positive process evidence can flag contention, but an empty process table
    # cannot replace missing utilisation readings when assessing a window.
    canary = witness.get('canaryInflation')
    readings = []
    for value in (before, after):
        if value is not None:
            readings.append(value)
    if isinstance(trajectory, list):
        readings.extend(value for value in trajectory if value is not None)
    if not readings and co_tenants is None and heartbeat is None and canary is None:
        return 'unclassifiable'
    peak = max(readings) if readings else None

    # Witness priority, in order of measured reliability:
    #   1. the AICore trajectory - in the controlled experiment it read 99 in all 9
    #      windows and labelled every one correctly;
    #   2. the driver's process table - reliable, but only says "a process exists";
    #   3. the canary - a POSITIVE-ONLY signal. Under that same 99-percent contention
    #      it exceeded its threshold in only 3 of 9 windows at one shape and 1 of 9 at
    #      the other, because it measures a 0.06 ms kernel whose own noise is the same
    #      order as the signal. It can therefore ADD a contention finding, but a
    #      canary that stayed flat is NOT evidence of a quiet window. Treating its
    #      silence as quiet would have mislabelled 6 to 8 of those 9 windows.
    if heartbeat or co_tenants:
        return 'contended'
    if peak is not None and peak >= CONTENDED_UTIL_PCT:
        return 'contended'
    if canary is not None and canary >= CANARY_INFLATION:
        return 'contended'
    # An empty process table or a flat canary cannot replace missing utilisation
    # readings: transient activity can be missed by either witness.
    if peak is None:
        return 'unclassifiable'
    # Sparse and flat is not the same as quiet. A window that was barely sampled
    # cannot claim the card was idle throughout; it is reported as unproven.
    rate = witness.get('samplingRateHz')
    if (peak is not None and rate is not None
            and rate < MIN_WITNESS_SAMPLES_PER_SECOND):
        return 'sparse'
    return 'quiet'


def _witness_summary(labelled):
    """How well-observed were the windows? Reported, never assumed.

    Contended and sparse windows are still USED - measured contention shifts the
    paired ratio by about 3 percent, which the paired design absorbs - but they are
    counted out loud. A verdict that rests on a quiet card and one that rests on an
    unexamined card must not look the same in the report.
    """
    rates = [(item.get('witness') or {}).get('samplingRateHz') for item in labelled]
    rates = [rate for rate in rates if rate is not None]
    canaries = [item['witness']['canaryInflation'] for item in labelled
                if (item.get('witness') or {}).get('canaryInflation') is not None]
    return {
        'quietWindows': sum(1 for item in labelled
                            if item['classification'] == 'quiet'),
        'contendedWindows': sum(1 for item in labelled
                                if item['classification'] == 'contended'),
        'sparseWindows': sum(1 for item in labelled
                             if item['classification'] == 'sparse'),
        'witnessSamplingRateHz': {
            'min': round(min(rates), 3) if rates else None,
            'median': round(statistics.median(rates), 3) if rates else None,
            'max': round(max(rates), 3) if rates else None,
            'note': ('npu-smi costs 0.5-3.2 s per invocation on this host, so this '
                     'rate is bounded by the tool, not by choice. It is reported so '
                     'a sparse "quiet" cannot be mistaken for an observed one.'),
        },
        'canaryInflation': {
            'min': round(min(canaries), 4) if canaries else None,
            'max': round(max(canaries), 4) if canaries else None,
            'windows': len(canaries),
            'threshold': CANARY_INFLATION,
        },
        # '' two kinds of unobserved: a window too sparsely sampled to claim quiet,
        # and a window that DID claim quiet while barely sampled.
        'quietWindowsWithSparseSampling': sum(
            1 for item in labelled
            if item['classification'] == 'quiet'
            and ((item.get('witness') or {}).get('samplingRateHz') or 0)
            < MIN_WITNESS_SAMPLES_PER_SECOND),
    }


def judge_timing(windows, shape):
    """The per-shape verdict: MATCH, FAIL or RETEST - never a silent average.

    speedup = vendor_ms / candidate_ms in the SAME window. The point estimate is
    the median of the per-window speedups and the interval is its paired
    bootstrap, because a verdict may not rest on one window.
    """
    labelled = [{**item, 'classification': classify_window(item.get('witness') or {})}
                for item in windows]
    usable = [item for item in labelled if item['classification'] != 'unclassifiable']
    void = len(labelled) - len(usable)
    witness_summary = _witness_summary(labelled)
    # ONLY quiet windows may enter the decision sample. Contended windows carry a
    # systematic shift, measured at 2.05x on an unchanged candidate and reversing
    # direction between shapes; averaging them in would move the median by the bias
    # rather than average it away.
    sample = [item for item in usable if item['classification'] == 'quiet']
    if not sample:
        return {'shape': shape, 'status': 'RETEST',
                'why': ('no quiet window was observed, so there is no unbiased sample '
                        'to decide on; contention was measured to shift the paired '
                        'ratio of an unchanged candidate by up to 2.05x, in a '
                        'direction that reverses between shapes'),
                'usableWindows': len(usable), 'voidWindows': void,
                'decisionSampleWindows': 0,
                **witness_summary,
                'windows': labelled}
    if len(usable) < MIN_WINDOWS:
        # Checked BEFORE the quiet-sample rules, and that order matters. sample is a
        # subset of usable, so once usable < MIN_WINDOWS the quiet rule would always
        # fire first and this branch would be unreachable - a guard that looks present
        # but never runs. This one carries the more precise reason: the windows could
        # not be labelled at all, which is a different failure from "they were labelled
        # and too many were contended".
        return {'shape': shape, 'status': 'RETEST',
                'why': 'only ' + str(len(usable)) + ' classifiable window(s) of a '
                       'required ' + str(MIN_WINDOWS) + '; the rest could not be '
                       'labelled, so no verdict is made in either direction',
                'usableWindows': len(usable), 'voidWindows': void,
                'decisionSampleWindows': len(sample),
                'quietFractionOfWindows': round(len(sample) / len(labelled), 4) if labelled else 0.0,
                **witness_summary,
                'windows': labelled}
    quiet_fraction = len(sample) / len(labelled) if labelled else 0.0
    if len(sample) < MIN_WINDOWS or quiet_fraction < MIN_QUIET_FRACTION:
        return {'shape': shape, 'status': 'RETEST',
                'why': (str(len(sample)) + ' quiet window(s) of ' + str(len(labelled))
                        + ' (' + str(round(100 * quiet_fraction, 1)) + '%) is not an '
                        'unbiased sample: ' + str(MIN_WINDOWS) + ' and '
                        + str(round(100 * MIN_QUIET_FRACTION)) + '% are both required, '
                        'because the windows excluded for contention are excluded '
                        'systematically, not at random'),
                'usableWindows': len(usable), 'voidWindows': void,
                'decisionSampleWindows': len(sample),
                'quietFractionOfWindows': round(quiet_fraction, 4),
                **witness_summary,
                'windows': labelled}
    speedups = sorted(item['speedup'] for item in sample)
    point = statistics.median(speedups)
    low, high = bootstrap_median_interval(speedups)
    p10 = speedups[max(0, int(0.10 * (len(speedups) - 1)))]
    p90 = speedups[min(len(speedups) - 1, int(0.90 * (len(speedups) - 1)))]
    spread = (p90 / p10) if p10 > 0 else math.inf
    if point >= MATCH_POINT and low >= MATCH_LOWER:
        status = 'EXCEED' if low > EXCEED_LOWER else 'MATCH'
    elif high < MATCH_POINT:
        # Even the optimistic end of the interval is slower than the vendor, so
        # noise cannot rescue this candidate: that is a FAIL, not a retest.
        status = 'FAIL'
    else:
        status = 'RETEST'
    reasons = []
    if status == 'RETEST':
        reasons.append('the paired evidence does not separate the candidate from the '
                       'threshold: point ' + str(number(point)) + ', interval ['
                       + str(number(low)) + ', ' + str(number(high)) + ']')
    if spread > MAX_RATIO_SPREAD:
        reasons.append('the windows disagree with each other (p90/p10 ' + str(number(spread))
                       + ' > ' + str(MAX_RATIO_SPREAD) + ')')
        if status in ('MATCH', 'EXCEED'):
            status = 'RETEST'
    return {
        'shape': shape, 'status': status,
        'pointSpeedup': number(point), 'interval95': [number(low), number(high)],
        'medianVendorMs': number(statistics.median(item['vendor_ms'] for item in usable)),
        'medianCandidateMs': number(statistics.median(item['candidate_ms'] for item in usable)),
        'usableWindows': len(usable), 'voidWindows': void,
        'decisionSampleWindows': len(sample),
        'quietFractionOfWindows': round(quiet_fraction, 4),
        **witness_summary,
        'ratioSpreadP90P10': number(spread),
        'reasons': reasons,
        'windows': labelled,
    }


# ═══════════════════════════════════════════════════════════════════════════
# D. the device-baseline audit: is a 'device time' actually device time?
# ═══════════════════════════════════════════════════════════════════════════


def audit_host_bound(workloads, aggregate=None):
    """Is the device baseline valid, judged per workload with the worst one binding?

    Two failure modes are checked, and they are different defects:

    * a workload whose host enqueue time is a large share of its device reading
      means that 'device time' is dispatch cost. The worst workload binds: a large
      shape cannot average a stalled small one away, which is precisely how a
      decode workload was once hidden behind a prefill measurement.

    * an archive that emits only an aggregate re-derives
      geomean(a_i)/geomean(b_i), which is identically the GEOMEAN of the
      per-workload ratios and never the maximum. For [0.04, 0.88] that re-derives
      0.1876 and passes an audit whose answer on the maximum is 'invalid'. The two
      verdicts are opposite, so an aggregate that disagrees with the per-workload
      maximum is itself a defect.
    """
    per = []
    for item in workloads:
        ratio = item.get('hostBoundRatio')
        if ratio is None:
            host, device = item.get('hostEnqueueMs'), item.get('deviceMedianMs')
            ratio = (host / device) if host and device else None
        per.append({'name': item.get('name'), 'ratio': ratio,
                    'hostEnqueueMs': item.get('hostEnqueueMs'),
                    'deviceMedianMs': item.get('deviceMedianMs')})
    measured = [item for item in per if item['ratio'] is not None]
    unmeasured = [item['name'] for item in per if item['ratio'] is None]
    max_ratio = max((item['ratio'] for item in measured), default=None)
    # geomean(host_i) / geomean(device_i) is IDENTICALLY the geometric mean of the
    # per-workload ratios, so it is computable from the ratios alone - and that
    # identity is the point. An archive that republishes only the host-side
    # aggregate necessarily re-derives THIS number, never the maximum.
    positive = [item['ratio'] for item in measured if item['ratio'] > 0]
    rederived = None
    if positive:
        rederived = math.exp(sum(math.log(value) for value in positive) / len(positive))
    reasons = []
    valid = True
    if not per:
        valid = False
        reasons.append('the baseline names no workload, so per-workload host-bound '
                       'ratios cannot be audited')
    if unmeasured:
        valid = False
        reasons.append('no enqueue clock was reported for ' + ', '.join(unmeasured)
                       + ', so the device reading of those workloads cannot be qualified')
    if max_ratio is not None and max_ratio >= HOST_BOUND_RATIO_WARN:
        valid = False
        worst = max(measured, key=lambda item: item['ratio'])
        reasons.append('workload ' + str(worst['name']) + ' is host-bound '
                       '(host_bound_ratio ' + str(number(worst['ratio'])) + ' at threshold '
                       + str(HOST_BOUND_RATIO_WARN) + '): its device time is host dispatch '
                       'cost, so a device baseline taken from it is invalid')
    inconsistent = None
    if aggregate is not None and aggregate.get('hostBoundRatio') is not None and max_ratio:
        declared = aggregate['hostBoundRatio']
        inconsistent = abs(declared - max_ratio) > max(1e-6, 1e-3 * max_ratio)
        if inconsistent:
            valid = False
            reasons.append('the emitted aggregate host_bound_ratio ('
                           + str(number(declared)) + ') is not the maximum over workloads ('
                           + str(number(max_ratio)) + '); an aggregate that disagrees with '
                           'the binding workload cannot qualify the baseline')
    if rederived is None or max_ratio is None:
        flips = False
    else:
        flips = ((rederived >= HOST_BOUND_RATIO_WARN)
                 != (max_ratio >= HOST_BOUND_RATIO_WARN))
    return {
        'valid': valid, 'reasons': reasons,
        'threshold': HOST_BOUND_RATIO_WARN,
        'workloads': per, 'maxRatio': number(max_ratio),
        'verdictOnMax': (None if max_ratio is None else
                         ('valid' if max_ratio < HOST_BOUND_RATIO_WARN else 'invalid')),
        'rederivedRatioOfGeomeans': number(rederived),
        'verdictOnRederived': (None if rederived is None else
                               ('valid' if rederived < HOST_BOUND_RATIO_WARN
                                else 'invalid')),
        'verdictFlips': flips,
        'aggregateDeclared': aggregate,
        'aggregateInconsistent': inconsistent,
    }



# ═══════════════════════════════════════════════════════════════════════════
# E. job evidence: missing marker, non-zero exit, timeout
# ═══════════════════════════════════════════════════════════════════════════


class JobEvidenceError(AcceptError):
    pass


def judge_job_evidence(record, stdout_text, checks=None):
    """Did the job that produced this artifact actually succeed?

    Every clause is a thing that has been seen to go wrong, and each one alone is
    enough to refuse: a run that timed out still has a log; a run that exited
    non-zero can still have printed a plausible JSON; a run that never reached the
    completion marker may have died after allocating its tensors. None of those is
    a pass, and none may be reclassified as a skip.
    """
    reasons = []
    if record.get('timedOut'):
        reasons.append('the job exceeded its deadline and was killed: a partial run '
                       'cannot report a measurement')
    if record.get('interrupted'):
        reasons.append('the job was interrupted: ' + str(record.get('reason') or 'no reason'))
    exit_code = record.get('exitCode')
    if exit_code != 0:
        reasons.append('the job exited ' + str(exit_code) + ', not 0')
    if not record.get('completionMarker'):
        reasons.append('the completion marker is absent from the job log, so the run '
                       'never reached the end of its task')
    if not stdout_text or not stdout_text.strip():
        reasons.append('the job produced no report')
    payload = parse_single_json(stdout_text)
    if payload is None:
        reasons.append('the job output is not one readable JSON object')
    size = len(stdout_text.encode('utf-8')) if stdout_text else 0
    if size > MAX_OUTPUT_BYTES:
        reasons.append('the job report is ' + str(size) + ' bytes, over the '
                       + str(MAX_OUTPUT_BYTES) + '-byte limit')
    collected = record.get('collected')
    if collected == 0:
        reasons.append('the job collected no tests or workloads at all')
    skipped = record.get('skipped')
    if collected and skipped is not None and skipped >= collected:
        reasons.append('every collected item was skipped, which is not a result')
    for name, value in sorted((checks or {}).items()):
        if value is not True:
            reasons.append('the observation check ' + name + ' is not true: ' + str(value))
    return {'verdict': 'pass' if not reasons else 'fail', 'reasons': reasons,
            'record': record, 'reportBytes': size, 'report': payload}


def parse_single_json(text):
    """The one JSON object a gate is allowed to print, or None."""
    if not text:
        return None
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def read_job_evidence(directory):
    """Load a recorded job: its process result, its stdout and its log."""
    directory = Path(directory)
    results = sorted(directory.rglob('process-result.json'))
    stdout = sorted(directory.rglob('stdout.log'))
    if not results:
        raise JobEvidenceError('no process-result.json under ' + str(directory))
    record = json.loads(results[0].read_text(encoding='utf-8'))
    text = stdout[0].read_text(encoding='utf-8', errors='replace') if stdout else ''
    marker_text = ''
    for path in sorted(directory.rglob('*.log')):
        try:
            marker_text += path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
    completion = bool(re.search(r'task=\S+ finished', marker_text))
    record = dict(record)
    record['completionMarker'] = completion
    record['completionMarkerDetail'] = ('found "task=<mode> finished" in a job log'
                                        if completion else 'absent')
    return {'record': record, 'stdout': text, 'source': str(results[0])}


# ═══════════════════════════════════════════════════════════════════════════
# F. the device child: one shape, empty cache, real work
# ═══════════════════════════════════════════════════════════════════════════


def child_main(args):
    """Compile the archived candidate from source and measure it.

    Runs in its own interpreter so the parent can bound it with a timeout, and so
    nothing the candidate module does at import time can reach the parent state.
    Every number it reports is measured here.
    """
    root = Path(args.candidate).resolve()
    # --shape is repeatable at the parent level, so argparse hands the child a
    # one-element list; the child measures exactly the one shape it was given.
    shape = args.shapes[0] if isinstance(args.shapes, list) else args.shape
    rows, cols = SHAPES[shape]
    cache = Path(os.environ['TILELANG_CACHE_DIR'])
    report = {'shape': shape, 'ok': False}

    def aibin_index():
        return {str(path.relative_to(cache)): digest_file(path)
                for path in sorted(cache.rglob('*.aibin'))}

    report['aibinBefore'] = aibin_index()
    if report['aibinBefore']:
        raise AcceptError('the acceptance cache ' + str(cache) + ' is not empty; a cold '
                          'recompile is the point of acceptance')

    try:
        import torch
        import torch_npu
    except BaseException as error:
        raise BlockedEnv('cannot import torch/torch_npu: ' + type(error).__name__ + ': '
                         + str(error)) from error
    if not torch.npu.is_available():
        raise BlockedEnv('no Ascend device is visible to the acceptance process')

    sys.path.insert(0, str(root))
    started = time.time()
    module = _load_candidate(root)
    report['candidateModule'] = str(getattr(module, '__file__', ''))
    report['importSeconds'] = round(time.time() - started, 2)

    w = torch.randn(cols, dtype=torch.bfloat16, device='npu')
    w_cpu = w.cpu()

    # ── precision, on all three held-out families ──────────────────────────
    precision_rows = []
    for family, x_cpu in _families(rows, cols).items():
        x = x_cpu.to('npu')
        truth, basis, why = _truth(x_cpu, w_cpu)
        if truth is None:
            precision_rows.append({'shape': shape, 'family': family, 'truth': basis,
                                   'why': why, 'candidate': None, 'vendor': None})
            continue
        vendor = torch_npu.npu_rms_norm(x, w, EPS)[0].cpu()
        candidate = _run_candidate(module, x, w).cpu()
        precision_rows.append({'shape': shape, 'family': family, 'truth': basis,
                               'vendor': _distance(vendor, truth),
                               'candidate': _distance(candidate, truth)})
        del x, vendor, candidate, truth
    report['precision'] = precision_rows

    # ── runtime witness: what kernels does the candidate launch? ───────────
    try:
        report['runtime'] = _witness(torch, torch_npu, module, rows, cols)
    except BaseException as error:
        report['runtime'] = {'error': type(error).__name__ + ': ' + str(error)[:300]}

    # ── paired timing ─────────────────────────────────────────────────────
    paired = _paired(torch, torch_npu, module, rows, cols, args.pairs,
                     args.repeat, args.neighbour_heartbeat)
    report['windows'] = paired['windows']
    report['pairedRun'] = {'runHeldSeconds': paired['runHeldSeconds'],
                           'device': paired['device'],
                           'pairs': args.pairs, 'repeat': args.repeat}
    report['aibinAfter'] = aibin_index()
    report['newAibin'] = sorted(set(report['aibinAfter']) - set(report['aibinBefore']))
    report['ok'] = True
    report['wallSeconds'] = round(time.time() - started, 2)
    Path(args.out).write_text(json.dumps(report, default=str), encoding='utf-8')
    return 0


def _load_candidate(root):
    """Import the archive entry module by path, not by package name."""
    entry = root / 'kernel.py'
    if not entry.is_file():
        candidates = sorted(root.glob('*.py'))
        if not candidates:
            raise AcceptError('the archive has no kernel.py and no other module: '
                              + str(root))
        entry = candidates[0]
    import importlib.util
    spec = importlib.util.spec_from_file_location('candidate_kernel', entry)
    if spec is None or spec.loader is None:
        raise AcceptError('cannot load ' + str(entry))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'run'):
        raise AcceptError('the archived candidate does not define run(x, w, eps): '
                          + str(entry))
    return module


def _run_candidate(module, x, w):
    """The candidate output under the contract: run(x, w, eps) -> y, bf16."""
    import torch
    with torch.no_grad():
        out = module.run(x, w, EPS)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise AcceptError('the candidate returned ' + type(out).__name__ + ', not a tensor')
    if tuple(out.shape) != tuple(x.shape):
        raise AcceptError('the candidate returned shape ' + str(tuple(out.shape))
                          + ', expected ' + str(tuple(x.shape)))
    return out


def _families(rows, cols):
    """The three input families, deterministic so every arm sees the same inputs."""
    import torch
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    normal = torch.randn(rows, cols, generator=generator,
                         dtype=torch.float32).to(torch.bfloat16)
    tiny = (torch.randn(rows, cols, generator=generator, dtype=torch.float32)
            * 1e-3).to(torch.bfloat16)
    outliers = torch.randn(rows, cols, generator=generator,
                           dtype=torch.float32).to(torch.bfloat16)
    outliers[:, 0] = 300.0
    return {'normal': normal, 'near_zero_rows': tiny, 'outlier_channels': outliers}


def _truth(x_cpu, w_cpu, chunk=512):
    """The definition in true float64, on the CPU, one rounding to bf16 at the end.

    The truth of THIS operator is a bf16 tensor: the contract rounds once, so
    comparing against an unrounded float64 result would charge every implementation
    a rounding it is required to make.
    """
    import torch
    try:
        xd, wd = x_cpu.double(), w_cpu.double()
        pieces = []
        for start in range(0, x_cpu.shape[0], chunk):
            block = xd[start:start + chunk]
            rstd = torch.rsqrt(block.pow(2).mean(-1, keepdim=True) + EPS)
            pieces.append((block * rstd * wd.unsqueeze(0)).to(torch.bfloat16))
        return torch.cat(pieces, dim=0), 'float64', ''
    except BaseException as error:
        return None, 'unavailable', type(error).__name__ + ': ' + str(error)[:200]


def _distance(out, truth):
    """Relative-L2, worst relative and mean absolute error, in CPU float64."""
    import torch
    y = out.double()
    t = truth.double()
    diff = (y - t).abs()
    denominator = t.abs().clamp_min(1e-6)
    numerator = float(torch.linalg.vector_norm((y - t).flatten()))
    scale = float(torch.linalg.vector_norm(t.flatten()))
    return {'rel_l2': (numerator / scale) if scale else float('inf'),
            'max_rel': float((diff / denominator).max()),
            'mean_abs': float(diff.mean()),
            'max_abs': float(diff.max())}


def _witness(torch, torch_npu, module, rows, cols):
    """Name the kernels the candidate actually launches, from a short profile."""
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device='npu')
    w = torch.randn(cols, dtype=torch.bfloat16, device='npu')
    _run_candidate(module, x, w)
    torch.npu.synchronize()
    names = _profiled_names(torch, torch_npu, lambda: _run_candidate(module, x, w))
    vendor = _profiled_names(torch, torch_npu,
                             lambda: torch_npu.npu_rms_norm(x, w, EPS)[0])
    vendor_names = [name for name in names
                    if re.search(r'rms.?norm', name, re.IGNORECASE)]
    return {
        'kernelNames': names,
        'vendorOperatorNames': vendor_names,
        'vendorOpObserved': bool(vendor_names),
        'launchedTilelangKernel': any(TILELANG_KERNEL_NAME.match(name) for name in names),
        'vendorRunKernelNames': vendor,
        'rule': ('the candidate must launch a TileLang-generated kernel and must not '
                 'launch the vendor fusion operator'),
    }


class _cd:
    """Change directory for the duration of a block (the profiler writes beside it)."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.previous = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.previous)
        return False


def _profiled_names(torch, torch_npu, fn, steps=3):
    """Exported chrome-trace event names from a short NPU profile of fn."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        with _cd(work):
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1,
                                                         repeat=1, skip_first=0),
                    on_trace_ready=lambda prof: prof.export_chrome_trace(
                        str(work / 'trace.json')),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False, data_simplification=False)) as prof:
                for _ in range(steps):
                    fn()
                    prof.step()
                torch.npu.synchronize()
        events = json.loads((work / 'trace.json').read_text(encoding='utf-8'))
    if isinstance(events, dict):
        events = events.get('traceEvents') or []
    return sorted({str(event.get('name')) for event in events
                   if isinstance(event, dict) and event.get('ph') == 'X'
                   and event.get('name')})


def _aicore_util(device):
    """The card AICore utilisation in percent right now, or None."""
    smi = os.environ.get('FK_ASCEND_SMI', 'npu-smi')
    try:
        done = subprocess.run([smi, 'info', '-t', 'usages', '-i', str(device)],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    # The per-device report must name the card it describes, or the answer is
    # about some other card and is reported as unknown rather than counted.
    if not re.search(r'NPU ID\s*:\s*' + re.escape(str(device)) + r'\s*$',
                     done.stdout, re.MULTILINE):
        return None
    match = re.search(r'Aicore Usage Rate\(%\)\s*:\s*([0-9]+)', done.stdout)
    return int(match.group(1)) if match else None


def _card_processes(device):
    """The PIDs resident on this card right now, or None when unreadable."""
    smi = os.environ.get('FK_ASCEND_SMI', 'npu-smi')
    try:
        done = subprocess.run([smi, 'info'], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return _parse_processes(done.stdout, device)


def _parse_processes(text, device):
    """The PIDs on this card, parsed out of one npu-smi listing.

    This is the honest co-tenant witness on a shared card. A heartbeat file needs a
    cooperating neighbour; a planted one would be a fake observation dressed as
    evidence. The device process table is read from the driver, cannot be authored
    by this script, and answers the question that actually matters for a window:
    did somebody else's process appear, vanish or stay resident while it ran.

    Split from the subprocess call so one npu-smi invocation can feed BOTH the
    utilisation reading and the process table. Two invocations would cost two
    seconds where one costs one, and sampling density is what this witness is short
    of.
    """
    pids, seen_table, saw_header = [], False, False
    for line in text.splitlines():
        if 'Process id' in line and 'Process name' in line:
            seen_table, saw_header = True, True
            continue
        if not seen_table:
            continue
        # The row separator closes the table. It must be the '-' rule, not the '='
        # rule: npu-smi puts '+====' immediately UNDER the header, so closing on it
        # would end the table before its first data row.
        if re.match(r'^\+(?:-+\+)+$', line):
            break
        if 'No running processes' in line:
            continue
        match = re.match(r'^\|\s*(\d+)\s*\|\s*(\d*)\s*\|', line)
        if match and match.group(2):
            if int(match.group(1)) == int(device):
                pids.append(int(match.group(2)))
    # No table at all means the report could not be read, NOT that the card is
    # clear. Returning [] here would turn a blind witness into a clean bill of
    # health, which is the one answer a witness must never invent.
    return sorted(set(pids)) if saw_header else None


def _card_snapshot(device):
    """Utilisation, HBM and the process table from ONE npu-smi call.

    One call, not two: a single invocation costs ~0.8 s on this host (measured), so
    splitting the readings across two commands doubles the cost of every sample for
    no extra information. The full listing already carries the per-card utilisation,
    the HBM figures and the device process table.

    Every field is None when unreadable, which is a different answer from a zero:
    None means the card cannot be described, 0 means it was described and was idle.
    Ownership is never reported - on a shared machine "what is on the card" is
    observable, "whose it is" is not.
    """
    smi = os.environ.get('FK_ASCEND_SMI', 'npu-smi')
    snapshot = {'utilPct': None, 'hbmUsedMb': None, 'hbmCapacityMb': None,
                'pids': None}
    try:
        done = subprocess.run([smi, 'info'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return snapshot
    if done.returncode != 0:
        return snapshot
    # A card occupies two lines, and the split is not obvious:
    #   | 1      | Ascend950PR | OK     | 270.0       64        0   / 0   |   <- power, temp only
    #   |        | 0000:55:00.0| 0     | 0      0   / 0    7192 / 131072  |   <- util, mem, HBM
    # Power is decimal and temperature is not, so matching "six integers in a row"
    # on either line fails. What IS regular: the second line carries the card's
    # "used / total" pairs, and the HBM pair is the last of them.
    lines = done.stdout.splitlines()
    for index, line in enumerate(lines):
        head = re.match(r'^\|\s*(\d+)\s*\|\s*([A-Za-z]\S*)\s*\|', line)
        if not head or int(head.group(1)) != int(device):
            continue
        if index + 1 >= len(lines):
            break
        detail = lines[index + 1]
        pairs = re.findall(r'(\d+)\s*/\s*(\d+)', detail)
        if not pairs:
            break
        snapshot['hbmUsedMb'] = int(pairs[-1][0])
        snapshot['hbmCapacityMb'] = int(pairs[-1][1])
        # The first integer on the detail line is NPU Util(%), and it precedes the
        # first pair from which the counters begin.
        util = re.match(r'^\|\s*\|\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)', detail)
        if util:
            snapshot['utilPct'] = int(util.group(1))
        break
    snapshot['pids'] = _parse_processes(done.stdout, device)
    if snapshot['utilPct'] is None and snapshot['hbmUsedMb'] is None:
        # The card row was never found, so nothing about this card was described -
        # including its process table, which would otherwise report "clear" for a
        # device that may not even exist.
        snapshot['pids'] = None
    return snapshot


class _CardSampler:
    """Polls the card while a window runs, so the window leaves a trajectory.

    A single utilisation reading before and after cannot distinguish "the card was
    idle" from "the card was busy in between". The trajectory can, and it measures
    actual contention rather than merely the presence of a neighbour.

    The cadence is NOT free to choose: one npu-smi invocation costs about 0.9 s on
    this host, measured, so a 0.2 s sample period is impossible - asking for one
    only makes the sampler fall behind and report a shorter trajectory than the
    window. The loop therefore samples back to back and records what it got, so the
    reported sample count is the truth rather than a target nobody met.
    """

    def __init__(self, device, interval=0.0):
        self.device = device
        self.interval = interval
        self.samples = []
        self.sampleCostSeconds = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._began = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exception):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20)
        return False

    def _loop(self):
        while not self._stop.is_set():
            started = time.time()
            snapshot = _card_snapshot(self.device)
            if self.sampleCostSeconds is None:
                self.sampleCostSeconds = round(time.time() - started, 3)
            self.samples.append({'atSeconds': round(time.time() - self._began, 3),
                                 'utilPct': snapshot['utilPct'],
                                 'hbmUsedMb': snapshot['hbmUsedMb'],
                                 'pids': snapshot['pids']})
            self._stop.wait(self.interval)

    def trajectory(self):
        utils = [item['utilPct'] for item in self.samples
                 if item['utilPct'] is not None]
        hbms = [item['hbmUsedMb'] for item in self.samples
                if item['hbmUsedMb'] is not None]
        pids = set()
        for item in self.samples:
            pids |= set(item['pids'] or [])
        return {
            'samples': len(self.samples),
            'sampleCostSeconds': self.sampleCostSeconds,
            'aicoreUtilSeries': [item['utilPct'] for item in self.samples],
            'aicoreUtilPeak': max(utils) if utils else None,
            'aicoreUtilMean': round(sum(utils) / len(utils), 2) if utils else None,
            'hbmUsedMbMin': min(hbms) if hbms else None,
            'hbmUsedMbMax': max(hbms) if hbms else None,
            'pidsSeen': sorted(pids) if any(item['pids'] is not None
                                            for item in self.samples) else None,
            'sampleSeries': list(self.samples),
        }


def _heartbeat(path):
    try:
        return int(Path(path).read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None


def _paired(torch, torch_npu, module, rows, cols, pairs, repeat, heartbeat_path):
    """A/B and B/A inside one window per sample, with a witness on every window."""
    from tilelang.profiler.msprof import bench_with_msprof
    device = str(os.environ.get('FK_ASCEND_DEVICE_ID')
                 or os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')).split(',')[0]
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device='npu')
    w = torch.randn(cols, dtype=torch.bfloat16, device='npu')
    # int32 on purpose: msprof own cache-flush kernel cannot lower an int8 vector
    # on Ascend ('SimtVF cannot convert vector type int8x8').
    cache = torch.empty(16 * 1024 * 1024 // 4, dtype=torch.int32, device='npu')
    candidate = lambda: _run_candidate(module, x, w)
    vendor = lambda: torch_npu.npu_rms_norm(x, w, EPS)[0]
    # A fixed slice of the SAME workload, re-measured every window. It is not part of
    # the verdict; it is the in-window busyness probe. Using the vendor op keeps it
    # independent of the candidate while staying the same kind of work (vector
    # units, HBM), so contention affects it the same way it affects the pair.
    canary_x = x[: min(64, rows)]
    canary_w = w
    canary = lambda: torch_npu.npu_rms_norm(canary_x, canary_w, EPS)[0]
    candidate()
    vendor()
    torch.npu.synchronize()
    # One sampler for the whole run, not one per window: a single npu-smi call
    # costs ~0.8 s here (measured), so a per-window sampler would spend more time
    # describing the card than measuring it, and its own overhead would land inside
    # the windows it is supposed to audit. Sampling across the run, with wall-clock
    # stamps, lets each window be labelled from the samples that bracketed it while
    # the sampling cost stays outside every window.
    windows = []
    run_started = time.time()
    with _CardSampler(device, interval=0.0) as sampler:
        for index in range(pairs):
            window_started = time.time()
            if index % 2 == 0:
                order = (('candidate', candidate), ('vendor', vendor))
            else:
                order = (('vendor', vendor), ('candidate', candidate))
            readings = {}
            for name, fn in order:
                readings[name] = float(bench_with_msprof(fn, cache, repeat))
            # The canary runs in the SAME window as the pair, on the same stream, so
            # an inflation in it is contention that overlapped this measurement -
            # observed, not inferred from a poll that could be seconds stale.
            canary_ms = float(bench_with_msprof(canary, cache, repeat))
            windows.append({
                'index': index,
                'order': [name for name, _ in order],
                'vendor_ms': round(readings['vendor'], 6),
                'candidate_ms': round(readings['candidate'], 6),
                'speedup': (readings['vendor'] / readings['candidate'])
                if readings['candidate'] else float('inf'),
                'canaryMs': round(canary_ms, 6),
                'startedAtSeconds': round(window_started - run_started, 3),
                'heldSeconds': round(time.time() - window_started, 3),
            })
        trajectory = sampler.trajectory()
        run_held = time.time() - run_started

    # Attach to each window the samples that bracketed it, and the canary reading
    # that ran INSIDE it.
    samples = trajectory.pop('sampleSeries', [])
    canary_values = sorted(window['canaryMs'] for window in windows)
    canary_base = statistics.median(canary_values) if canary_values else None
    for window in windows:
        begin = window['startedAtSeconds']
        end = begin + window['heldSeconds']
        bracketing = [item for item in samples
                      if begin - 1.5 <= item['atSeconds'] <= end + 1.5]
        utils = [item['utilPct'] for item in bracketing if item['utilPct'] is not None]
        pids = set()
        for item in bracketing:
            pids |= set(item['pids'] or [])
        churn = sorted(pids - {os.getpid()}) if any(
            item['pids'] is not None for item in bracketing) else None
        inflation = (window['canaryMs'] / canary_base
                     if canary_base and window.get('canaryMs') else None)
        window['samplingRateHz'] = (round(len(bracketing) / window['heldSeconds'], 3)
                                    if window['heldSeconds'] else None)
        window['witness'] = {
            'aicoreUtilBefore': utils[0] if utils else None,
            'aicoreUtilAfter': utils[-1] if utils else None,
            'aicoreUtilTrajectory': utils,
            'aicoreUtilPeak': max(utils) if utils else None,
            # The driver's own process table: a co-tenant witness this script
            # cannot author. Empty means the card was described and clear; None
            # means it could not be described at all.
            'coTenants': churn,
            'bracketingSamples': len(bracketing),
            # Contention measured at kernel granularity, inside this window.
            'canaryMs': window.get('canaryMs'),
            'canaryInflation': round(inflation, 4) if inflation else None,
            # Lives in the witness because that is what classify_window reads. Stored
            # only at the top level it would be reported but never consulted, and the
            # sparse-window guard would silently never fire.
            'samplingRateHz': window['samplingRateHz'],
        }
    return {'windows': windows,
            'runHeldSeconds': round(run_held, 3),
            'device': device,
            'canaryBaselineMs': round(canary_base, 6) if canary_base else None,
            'trajectory': {**trajectory, 'sampleSeries': samples}}


# ═══════════════════════════════════════════════════════════════════════════
# G. orchestration
# ═══════════════════════════════════════════════════════════════════════════


def run_child(script, args, cache, timeout):
    """Run one shape device child in a fresh interpreter and return its report."""
    out = cache.parent / ('report-' + args.shape + '.json')
    if out.exists():
        out.unlink()
    log = cache.parent / ('child-' + args.shape + '.log')
    environment = dict(os.environ)
    environment['TILELANG_CACHE_DIR'] = str(cache)
    environment.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    environment['PYTHONUNBUFFERED'] = '1'
    command = [sys.executable, '-B', str(script), '--_device-child',
               '--candidate', str(args.candidate), '--shape', args.shape,
               '--pairs', str(args.pairs), '--repeat', str(args.repeat),
               '--out', str(out)]
    if args.neighbour_heartbeat:
        command += ['--neighbour-heartbeat', str(args.neighbour_heartbeat)]
    if args.device is not None:
        environment['ASCEND_RT_VISIBLE_DEVICES'] = str(args.device)
        environment['FK_ASCEND_DEVICE_ID'] = str(args.device)
    started = time.time()
    try:
        with log.open('wb') as sink:
            done = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT,
                                  env=environment, timeout=timeout)
        code = done.returncode
    except subprocess.TimeoutExpired:
        return {'shape': args.shape, 'ok': False, 'timedOut': True, 'log': str(log),
                'wallSeconds': round(time.time() - started, 1),
                'why': 'the device child exceeded ' + str(int(timeout)) + 's and was killed'}
    elapsed = round(time.time() - started, 1)
    if not out.is_file():
        tail = log.read_text(encoding='utf-8', errors='replace')[-1500:] if log.is_file() else ''
        return {'shape': args.shape, 'ok': False, 'exitCode': code, 'log': str(log),
                'wallSeconds': elapsed, 'stderrTail': tail,
                'why': 'the device child produced no report'}
    report = json.loads(out.read_text(encoding='utf-8'))
    report['wallSeconds'] = elapsed
    report['exitCode'] = code
    report['log'] = str(log)
    return report


def orchestrate(args):
    """Run every stage, judge each one, and publish one JSON object."""
    root = Path(args.candidate).resolve()
    if not root.is_dir():
        raise AcceptError('candidate artifact is not a directory: ' + str(root))
    started = time.time()
    result = {'schema': SCHEMA, 'protocol': PROTOCOL,
              'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'candidate': {'root': str(root), 'shapes': list(args.shapes)}}

    scan, artifact = scan_artifact(root)
    result['candidate']['files'] = artifact['files']

    if args.job_evidence:
        evidence = read_job_evidence(Path(args.job_evidence))
        job = judge_job_evidence(evidence['record'], evidence['stdout'],
                                 evidence.get('checks'))
        job['source'] = evidence['source']
        result['job'] = job

    if args.baseline_json:
        archive = json.loads(Path(args.baseline_json).read_text(encoding='utf-8'))
        audit = audit_host_bound(_archive_workloads(archive),
                                 _archive_aggregate(archive))
        audit['archive'] = str(args.baseline_json)
        result['hostBoundAudit'] = audit
    elif args.workload_ratio:
        aggregate = (None if args.aggregate_ratio is None
                     else {'hostBoundRatio': args.aggregate_ratio})
        result['hostBoundAudit'] = audit_host_bound(_ratio_workloads(args.workload_ratio),
                                                    aggregate)

    if args.check_only:
        result['authenticity'] = judge_authenticity(scan, artifact, None, None)
        result['precision'] = None
        result['timing'] = None
        result['verdict'] = decide(result, check_only=True)
        result['elapsedSeconds'] = round(time.time() - started, 1)
        return result

    children = {}
    for shape in args.shapes:
        cache = Path(args.cache) / ('accept-' + shape + '-' + str(os.getpid()))
        if cache.exists() and any(cache.iterdir()):
            raise AcceptError('the acceptance cache ' + str(cache) + ' is not empty; '
                              'acceptance recompiles from empty')
        cache.mkdir(parents=True, exist_ok=True)
        child_args = argparse.Namespace(**vars(args))
        child_args.shape = shape
        child = run_child(Path(__file__).resolve(), child_args, cache, args.child_timeout)
        if not child.get('ok'):
            detail = str(child.get('why') or child.get('stderrTail') or 'no report')[:700]
            if child.get('timedOut'):
                raise BlockedEnv(detail)
            raise BlockedEnv('the device child failed: ' + detail)
        children[shape] = child

    compile_evidence = {
        'coldCache': True,
        'aibinCount': sum(len(child.get('newAibin') or []) for child in children.values()),
        'shapes': {shape: {'newAibin': child.get('newAibin'),
                           'module': child.get('candidateModule'),
                           'importSeconds': child.get('importSeconds')}
                   for shape, child in children.items()},
    }
    runtime = next((child.get('runtime') for child in children.values()
                    if child.get('runtime')), None)
    result['authenticity'] = judge_authenticity(scan, artifact, runtime, compile_evidence)

    rows = [row for child in children.values() for row in (child.get('precision') or [])
            if row.get('candidate')]
    result['precision'] = judge_precision(rows)
    result['timing'] = {shape: judge_timing(child.get('windows') or [], shape)
                        for shape, child in children.items()}
    for shape, child in children.items():
        if shape in result['timing']:
            result['timing'][shape]['canaryBaselineMs'] = child.get('canaryBaselineMs')
            result['timing'][shape]['cardTrajectory'] = {
                'sampleCostSeconds': (child.get('trajectory') or {}).get(
                    'sampleCostSeconds'),
                'samples': len((child.get('trajectory') or {}).get('sampleSeries') or []),
            }
    result['measurement'] = {
        'basis': ('msprof per-kernel device time '
                  '(tilelang.profiler.msprof.bench_with_msprof); host-free'),
        'pairsRequested': args.pairs, 'repeatPerWindow': args.repeat,
        'device': str(args.device if args.device is not None
                      else os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unset')),
        'childWallSeconds': {shape: child.get('wallSeconds')
                             for shape, child in children.items()},
        'coldCompile': True,
    }
    result['verdict'] = decide(result, check_only=False)
    result['elapsedSeconds'] = round(time.time() - started, 1)
    return result


def _archive_workloads(archive):
    performance = archive.get('performance') or archive
    return [{'name': row.get('name'),
             'hostBoundRatio': row.get('host_bound_ratio'),
             'hostEnqueueMs': row.get('host_enqueue_ms'),
             'deviceMedianMs': row.get('device_median_ms')}
            for row in (performance.get('workloads') or [])]


def _archive_aggregate(archive):
    performance = archive.get('performance') or archive
    return {'hostBoundRatio': performance.get('host_bound_ratio')}


def _ratio_workloads(specs):
    """'decode=0.88,prefill=0.04' -> workload rows, for auditing a supplied case."""
    rows = []
    for spec in specs:
        for item in str(spec).split(','):
            if not item.strip():
                continue
            name, _, value = item.partition('=')
            rows.append({'name': name.strip() or ('workload' + str(len(rows))),
                         'hostBoundRatio': float(value)})
    return rows


def decide(result, check_only):
    """The single verdict, from the judged sections and nothing else."""
    authenticity = result.get('authenticity') or {}
    if authenticity.get('verdict') != 'pass':
        return {'status': 'FAIL', 'stage': 'authenticity',
                'reasons': authenticity.get('reasons')
                or ['the candidate was not shown to compute in its own kernel']}
    if result.get('job') is not None and result['job'].get('verdict') != 'pass':
        return {'status': 'FAIL', 'stage': 'job', 'reasons': result['job'].get('reasons')}
    baseline = result.get('hostBoundAudit')
    if baseline is not None and not baseline.get('valid'):
        return {'status': 'INVALID_EVIDENCE', 'stage': 'deviceBaseline',
                'reasons': baseline.get('reasons')}
    if check_only:
        return {'status': 'CHECKS_ONLY', 'stage': 'static',
                'reasons': ['this run judged no device measurement']}
    precision = result.get('precision') or {}
    if precision.get('verdict') != 'pass':
        return {'status': 'FAIL', 'stage': 'precision',
                'reasons': precision.get('failures') or ['no precision row was measured']}
    timing = result.get('timing') or {}
    failures, retests = [], []
    for shape, block in timing.items():
        status = block.get('status')
        if status == 'FAIL':
            failures.append(shape + ': ' + str(block.get('pointSpeedup')) + 'x with '
                            'interval ' + json.dumps(block.get('interval95'))
                            + '; the vendor is faster at every plausible paired reading')
        elif status == 'RETEST':
            retests.append(shape + ': ' + '; '.join(block.get('reasons')
                                                    or ['the paired evidence is '
                                                        'inconclusive']))
    if failures:
        return {'status': 'FAIL', 'stage': 'performance', 'reasons': failures,
                'bestCorrectArtifact': str((result.get('candidate') or {}).get('root')),
                'retests': retests}
    if retests:
        return {'status': 'RETEST', 'stage': 'performance', 'reasons': retests}
    exceeded = all(block.get('status') == 'EXCEED' for block in timing.values())
    return {'status': 'EXCEED' if exceeded else 'MATCH', 'stage': 'performance',
            'reasons': [],
            'perShape': {shape: {'status': block.get('status'),
                                 'pointSpeedup': block.get('pointSpeedup'),
                                 'interval95': block.get('interval95')}
                         for shape, block in timing.items()}}


# The keys a decided report cannot lose: the samples and the verdict they support.
# Everything else is a summary of evidence recorded in full elsewhere.
ESSENTIAL_KEYS = frozenset({'schema', 'protocol', 'verdict', 'timing', 'precision',
                            'authenticity'})


def emit(result):
    """One JSON object on stdout, inside the evaluator 64 KiB limit."""
    limit = MAX_OUTPUT_BYTES
    dropped = []
    text = json.dumps(result, sort_keys=True, default=str)
    if len(text.encode('utf-8')) > limit:
        # Drop redundant per-window fields first, then the least load-bearing
        # detail. The raw samples, the intervals and the verdict never go.
        for block in (result.get('timing') or {}).values():
            for window in block.get('windows') or []:
                for key in ('order', 'index'):
                    window.pop(key, None)
        dropped.extend(['order', 'index'])
        text = json.dumps(result, sort_keys=True, default=str)
    if len(text.encode('utf-8')) > limit:
        for block in (result.get('timing') or {}).values():
            for window in block.get('windows') or []:
                window.pop('witness', None)
        dropped.append('witness')
        text = json.dumps(result, sort_keys=True, default=str)
    if len(text.encode('utf-8')) > limit:
        (result.get('authenticity') or {}).pop('runtimeWitness', None)
        dropped.append('runtimeWitness')
        text = json.dumps(result, sort_keys=True, default=str)
    # Then the evidence payloads, largest first. Each summarises something recorded
    # elsewhere in full (the receipt, the archive, the job log); the paired samples
    # exist only in this object, so they are never shed.
    for container, key in (((result.get('job') or {}), 'report'),
                           ((result.get('job') or {}), 'record'),
                           ((result.get('authenticity') or {}).get('static')
                            or {}, 'imports'),
                           ((result.get('hostBoundAudit') or {}),
                            'aggregateDeclared')):
        if len(text.encode('utf-8')) <= limit:
            break
        if key in container:
            container.pop(key, None)
            dropped.append(key)
            text = json.dumps(result, sort_keys=True, default=str)
    # Finally, every top-level key outside the contract. A key this script did not
    # define is not part of the report, so it is the first thing an oversized
    # object can afford to lose - and dropping by WHITELIST rather than by a known
    # blacklist means a field added later cannot quietly push the object over the
    # limit.
    if len(text.encode('utf-8')) > limit:
        for key in sorted(set(result) - ESSENTIAL_KEYS):
            result.pop(key, None)
            dropped.append(key)
            text = json.dumps(result, sort_keys=True, default=str)
            if len(text.encode('utf-8')) <= limit:
                break
    if len(text.encode('utf-8')) > limit:
        # Nothing droppable is left that would not take the samples or the verdict
        # with it. Refusing to print an oversized object is the honest failure: a
        # report the evaluator cannot read is not a decided gate.
        raise AcceptError('the report cannot be expressed within ' + str(limit)
                          + ' bytes without dropping the samples or the verdict')
    if dropped:
        result['sizeGuard'] = {'limitBytes': limit,
                               'bytes': len(text.encode('utf-8')),
                               'dropped': sorted(set(dropped))}
        text = json.dumps(result, sort_keys=True, default=str)
        if len(text.encode('utf-8')) > limit:
            result.pop('sizeGuard', None)
            text = json.dumps(result, sort_keys=True, default=str)
    sys.stdout.write(text + chr(10))


STATUS_EXITS = {'MATCH': 0, 'EXCEED': 0, 'CHECKS_ONLY': 0, 'RETEST': 4}


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROG, description='independent acceptance for a BF16 RMSNorm candidate')
    parser.add_argument('--candidate', required=True,
                        help='archived candidate directory holding kernel.py with run()')
    parser.add_argument('--shape', action='append', dest='shapes', default=None,
                        help='shape to accept (repeatable); default both frozen shapes')
    parser.add_argument('--pairs', type=int, default=12,
                        help='paired windows per shape (default 12; at least '
                             + str(MIN_WINDOWS) + ' decide anything)')
    parser.add_argument('--repeat', type=int, default=8,
                        help='msprof repeats inside one window (default 8)')
    parser.add_argument('--cache', default=None,
                        help='parent of the fresh kernel caches acceptance compiles into')
    parser.add_argument('--device', default=None,
                        help='physical device id this run is authorized to use')
    parser.add_argument('--child-timeout', type=float, default=115.0,
                        help='seconds one shape may take on the device (default 115)')
    parser.add_argument('--job-evidence', default=None,
                        help='directory holding process-result.json and stdout.log')
    parser.add_argument('--baseline-json', default=None,
                        help='an archived reference report, audited for host-bound '
                             'device time')
    parser.add_argument('--workload-ratio', action='append', default=None,
                        help='name=ratio[,name=ratio] to audit directly')
    parser.add_argument('--aggregate-ratio', type=float, default=None,
                        help='the aggregate host_bound_ratio an archive declared')
    parser.add_argument('--check-only', action='store_true',
                        help='judge the archive and job evidence without touching a device')
    parser.add_argument('--neighbour-heartbeat', default=None,
                        help='a co-tenant heartbeat file, read around every window')
    parser.add_argument('--_device-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--out', default=None, help=argparse.SUPPRESS)
    return parser


def main(argv):
    args = build_parser().parse_args(argv)
    if args._device_child:
        return child_main(args)
    args.shapes = list(args.shapes) if args.shapes else list(SHAPES)
    unknown = [name for name in args.shapes if name not in SHAPES]
    if unknown:
        raise AcceptError('unknown shape(s) ' + ', '.join(unknown) + '; the frozen set is '
                          + ', '.join(SHAPES))
    if not args.check_only and args.cache is None:
        raise AcceptError('--cache names where acceptance compiles; it is required for a '
                          'device run and must start empty')
    if not args.check_only and args.pairs < MIN_WINDOWS:
        raise AcceptError('--pairs must be at least ' + str(MIN_WINDOWS) + '; fewer '
                          'windows cannot support a verdict in either direction')
    result = orchestrate(args)
    emit(result)
    return STATUS_EXITS.get(result['verdict']['status'], 1)


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BlockedEnv as error:
        sys.stderr.write(PROG + ': ' + str(error) + chr(10))
        raise SystemExit(75)
    except AcceptError as error:
        sys.stderr.write(PROG + ': ' + str(error) + chr(10))
        raise SystemExit(error.EXIT)



