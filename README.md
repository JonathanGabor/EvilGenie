# Reward Hacking Benchmark

## Installation
```
uv venv
source .venv/bin/activate
uv sync
```
Set API keys
```
export OPENAI_API_KEY=<your openai api key>
export ANTHROPIC_API_KEY=<your anthropic api key>
export GOOGLE_API_KEY=<your google api key>
```

## Example Usage

```
python src/run_agent_tests.py --agents claude codex --difficulty hard --release-version v6 --max-problems 2 --no-cleanup
```
