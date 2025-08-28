# Reward Hacking Benchmark

## Installation
```
uv venv
source .venv/bin/activate
uv sync
```
Install cli based agents
```
brew install codex
brew install gemini-cli
```
Set API keys

Option A: .env file (recommended)
```
cp .env.example .env
# then edit .env and add your keys
```
The CLI loads `.env` automatically at startup.

Option B: export in shell
```
export OPENAI_API_KEY=<your openai api key>
export ANTHROPIC_API_KEY=<your anthropic api key>
export GOOGLE_API_KEY=<your google api key>
```

## Example Usage

```
python src/run_agent_tests.py --agent claude --difficulty medium --release-version v6 --random --no-cleanup
```
