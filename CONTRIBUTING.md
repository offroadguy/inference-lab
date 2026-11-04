# Contributing

Start with the free demo in the README. Python 3.12 and uv are required.
Native macOS and the Linux CI workflow are supported targets; native Windows
is not supported because the journal lock uses `fcntl`. Use WSL2 for Linux behavior;
that path has not been validated by the maintainer.

Before opening a pull request:

```sh
uv sync --frozen
uv run ruff check .
uv run pytest -q
```

Tests and the simulator need no cloud credentials. Never add a paid cloud run to CI.
For measurement changes, explain the denominator, observation window, treatment of
errors and warm-up, and provide a deterministic test with a known expected result.
For UI changes, check both an empty journal and simulated results.

Open an issue with the commit, OS/Python version, command, expected behavior and
redacted error. Do not attach `.env`, tokens, authenticated endpoint URLs, private
prompts, or complete journals without inspecting them. Keep contributions focused;
include screenshots for interface changes and mark simulated results clearly.

Do not claim capacity, cost savings or model quality from a single short smoke run.
Contributions are submitted under the project's Apache-2.0 license.
