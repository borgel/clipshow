#!/usr/bin/env bash
set -euo pipefail

gemini -m gemini-2.5-pro -p "You are a senior test engineer auditing a Python project for test quality. Your job is to find problems, not praise what works.

## Instructions

1. Read PLAN.md to understand the full architecture and build order.
2. Read every test file in tests/.
3. Read every source file in clipshow/ that has corresponding tests.
4. For each test file, check ALL of the following:

### Coverage Completeness
- Does every public function/method in the source have at least one test?
- Are error paths tested (exceptions, edge cases, empty inputs)?
- Are boundary conditions tested (zero, negative, max values)?

### Short-Circuited Tests (CRITICAL)
- Tests that assert True or pass without doing real work
- Tests that only check types but not values
- Tests with overly loose assertions (approx with huge tolerances)
- Tests that mock so aggressively they test the mock, not the code
- Tests that catch exceptions too broadly (bare except, catching Exception)
- Assertions that can never fail (e.g., asserting a list is a list)

### Bypassed / Cheated Tests
- Tests marked @pytest.skip or xfail without good reason
- Tests with TODO/FIXME comments indicating incomplete work
- Tests that return early before meaningful assertions
- Conditional skips that always trigger

### Timing and Async Gaps
- Race conditions in threaded/Qt signal tests
- Missing waitSignal/waitUntil for async operations
- Hard-coded sleeps instead of proper synchronization
- Tests that pass due to timing luck rather than correctness

### Architecture Gaps (vs PLAN.md)
- Modules described in the plan that have no tests at all
- Features described in the plan that aren't tested
- Integration points between modules that aren't tested

## Output Format

For each issue found, output:
- **File**: the test file (and line number if possible)
- **Severity**: CRITICAL / HIGH / MEDIUM / LOW
- **Category**: one of the categories above
- **Issue**: what's wrong
- **Fix**: what should be done

End with a summary count: N critical, N high, N medium, N low.
If no issues are found, say 'AUDIT PASSED: No issues found.'"
