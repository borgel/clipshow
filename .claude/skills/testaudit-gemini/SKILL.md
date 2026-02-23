---
name: testaudit-gemini
description: Audit test coverage completeness using Gemini. Checks for short-circuited tests, missing coverage, bypassed assertions, and gaps vs the architecture plan.
context: fork
---

Run the script `scripts/audit.sh` to get a detailed audit of test quality and coverage.

Review the output and present the findings to the user, organized by severity.
