# BLOCKER — Phase 0 (Lightning port): SNFO template is absent

**Status:** BLOCKED at the §0.5 reference fence during orient, *before* Phase 0 could
begin. No smoke test was run (the fence failed first), so there is **no `.out`/`.err`
tail to attach** — the failure is a missing load-bearing reference, not a failed run.

## What failed

The whole port is defined as "restructure S2S to **mirror SNFO** so that S2S and SNFO
share one codebase whose only material difference is the model definition." SNFO is the
template for **every** phase. The driver's §0.5 reference fence requires the SNFO
template to be **present and readable**:

```
/home/rmeht/Projects/SNFO/{train.py, modules/train_module.py, data/datamodule.py,
                           common/bench_callback.py, environment.yml}
```

All five are **absent**. The path belongs to a different account (`rmeht`); this session
runs as `mehta5`, and `/home/rmeht` does not exist.

## Exact errors / evidence (verified four independent ways)

```
$ whoami → mehta5   ;   HOME=/home/mehta5
$ ls /home/rmeht/            → ls: cannot access '/home/rmeht/': No such file or directory
$ ls /home/rmeht/Projects/   → ls: cannot access '/home/rmeht/Projects/': No such file or directory
```

1. **Not at the documented path** `/home/rmeht/Projects/SNFO` — `rmeht` home does not exist.
2. **Not a sibling of S2S** — `/project/pedramh/shared/` contains only
   `anthonyz/ conda/ cuda-profiling/ profiling-env/ README S2S/`. No `SNFO`.
3. **Not under `/project/pedramh/`** — no `SNFO` entry. (There **is** `/project/pedramh/SFNO`,
   but that is **SFNO ≠ SNFO**: a `modulus-makani` checkout owned by `zand`, with **none** of
   the SNFO Lightning-template files — `train.py`, `val.py`, `modules/train_module.py`,
   `data/datamodule.py`, `common/bench_callback.py`, `environment.yml`, `modules/models` all
   absent. Ruled out; not substituted.)
4. **No SNFO source to clone** — git remotes are only `myfork → rmehta1987/S2S` and
   `origin → masak1112/S2S`; both are **S2S**. The driver/agents give SNFO as a filesystem
   path, never a clone URL. Memory holds no SNFO reference. So this cannot be self-resolved
   from within the session.

## Why no phase was attempted (central discipline: verify, never assume)

Every phase mirrors SNFO and is therefore unverifiable without it:

- Phase 0 scaffold = "SNFO-style dirs ... with `__init__.py` where SNFO has them" → requires
  SNFO's actual structure.
- Phases 1–4 = mirror `data/datamodule.py` / `modules/train_module.py` / `train.py` / `val.py`.
- Phase 5 = "reconcile **with SNFO** so only the model definition differs" → impossible to
  verify against a template that is not present.
- Phase 6 = cross-reference integrity → nothing upstream exists.

The porter agent's mapping table claims "exact, verified paths/symbols" under
`/home/rmeht/Projects/SNFO/`, but those are now **unverifiable**. §0.5 is explicit: when a
load-bearing reference is absent, **STOP — do not proceed from memory**. Scaffolding/porting
from the agent's secondhand description would advance the green floor on something that
cannot be checked against its spec, and would make Phase 5 reconciliation meaningless. No
**independent** phase exists (every phase depends on SNFO), so per §9 the loop emits BLOCKED
rather than continuing.

## Secondary finding (not the primary blocker)

The driver mandates "Read `CLAUDE.md` FIRST," but **no `CLAUDE.md` exists** in the repo
(not at root, not git-tracked, not within `maxdepth 3`). The driver restates its hard
constraints inline, and `CLAUDE.md` is not part of the §0.5 fence, so this did not by itself
block — but it is reported for fidelity. If the operator expects a `CLAUDE.md`, it is missing.

## What the operator must provide to unblock

1. **Make SNFO present and readable** for user `mehta5` — either:
   - place/clone it at `/home/rmeht/Projects/SNFO/` (the path the porter + driver hard-code), or
   - give the **real reachable path** to SNFO on this node (so the porter/driver references can
     be repointed), or
   - give the **SNFO git clone URL** (this driver runs on `--partition=build`, which has egress).
2. **Confirm `LPORT_ENV`** — the driver leaves it as a `<<SET ME>>` placeholder. The default
   Phase-0 approach ("clone SNFO's `environment.yml`") also depends on SNFO being present.
3. Optionally confirm whether a `CLAUDE.md` is expected in this repo.

Once SNFO is reachable, re-run orient (§1); the reference fence will pass and Phase 0 can begin.
