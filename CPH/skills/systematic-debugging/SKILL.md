---
name: systematic-debugging
description: Root-cause debugging protocol that eliminates guesswork. Enforces hypothesis testing, minimal reproducible test cases, traceback analysis, and regression prevention.
---

# Systematic Debugging & Root Cause Analysis

This skill provides a structured methodology for diagnosing and resolving bugs, failures, and test breakages without resorting to speculative trial-and-error ("shotgun debugging").

---

## 1. The Core Law of Debugging

> [!IMPORTANT]
> **Never modify code based on a guess.**
> Every code change made to fix a bug must be backed by a verified hypothesis explaining exactly why the failure occurred and why the fix resolves it.

---

## 2. The 5-Step Diagnostic Protocol

```text
[1. Reproduce] ──> [2. Isolate] ──> [3. Hypothesize] ──> [4. Prove/Disprove] ──> [5. Fix & Prevent]
```

### Step 1: Reproduce
* Run the exact failing command or script to witness the failure firsthand.
* If the failure occurs in a complex pipeline, construct a minimal, standalone reproduction script (e.g. in `scratch/repro.py`) that isolates the failing function with mock or sample inputs.

### Step 2: Isolate Failure Locus
* **Traceback Examination**:
  - Locate the exact file and line number where the exception is raised.
  - Inspect the call stack frames leading up to the failure.
* **State Inspection**:
  - What were the runtime types and values of the arguments passed to the failing function?
  - Was a dictionary missing a key (`KeyError`)? Was a variable `None` (`AttributeError`)? Was an encoding or byte order mismatched?
* Use `view_file` around the exact line number.

### Step 3: Formulate Falsifiable Hypotheses
* Formulate 1–2 explicit hypotheses:
  - *"Hypothesis A: The EXIF reader expects a file path string, but received a raw bytes buffer, causing `os.path.exists()` to throw a `TypeError`."*
  - *"Hypothesis B: The perceptual hash function expects RGB channels, but the input image has an RGBA alpha channel."*

### Step 4: Prove or Disprove the Hypothesis
* Add a temporary debug log, assertion, or run an isolated one-line command to inspect the actual value.
* Verify whether the hypothesis holds true before writing the fix.

### Step 5: Surgical Fix & Regression Guard
* Implement the minimal fix addressing the root cause (not just masking the symptom).
* Re-run the reproduction script to prove the fix works.
* Re-run the broader test suite to ensure no regressions were introduced elsewhere.
* Clean up any temporary debug prints or scratch scripts.

---

## 3. Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | What to Do Instead |
| :--- | :--- | :--- |
| **Shotgun Debugging** | Tweaking random lines hoping tests pass. | Stop. Trace the data flow and identify the exact failing invariant. |
| **Silent Swallowing** | Wrapping code in `try: ... except Exception: pass`. | Handle specific exceptions, log meaningful context, and fail fast. |
| **Symptom Patching** | Adding `if x is None: return ""` without asking why `x` was `None`. | Trace upstream to understand why `x` became `None`. |
| **Blind Refactoring** | Rewriting the entire function when 1 condition was inverted. | Fix only the broken condition with a targeted one-line edit. |
