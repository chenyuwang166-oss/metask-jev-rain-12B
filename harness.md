# Evaluator: official TypeSafe adapter

Install the official JevBench harness in its own environment. Its source is not
vendored here. Use the supplied official clone revision bb05a33 (2026-09-29).
The harness and service must run on the same machine (loopback binding).
Start `serve.sh` in the model environment, wait for the printed `/health` result,
then run the official harness from its environment:

```bash
python -m jevbench.cli run --adapter typesafe \
  --endpoint http://127.0.0.1:8000 --model metask-jev-rain-12B --key-env '' \
  --tasks "$TASKS_JSONL" --results "$RUN_DIR/results.jsonl" \
  --ledger "$RUN_DIR/ledger.jsonl" --raw-dir "$RUN_DIR/raw" \
  --manifest "$RUN_DIR/manifest.json" --cap-usd 15 \
  --price-in-per-m 0.05 --price-out-per-m 0.15
python -m jevbench.cli summarize --tasks "$TASKS_JSONL" \
  --results "$RUN_DIR/results.jsonl" --ledger "$RUN_DIR/ledger.jsonl"
```

Set TASKS_JSONL and a new RUN_DIR explicitly. Override 8000 consistently if using
another port. These options match the supplied official CLI: `typesafe` appends
`/v1/systemone`, sends `state`, `model`, `questions.decision`, and reads typed
probabilities and `usage`. Empty `--key-env ''` disables token authentication.
The quoted empty argument is intentional (Bash). Never add `/v1/systemone` to
the base endpoint. All output locations are explicit rather than harness defaults.

The prices are fixed 12B reference assumptions in USD per million tokens:
0.05 input / 0.15 output. The nearest listed size class is google/gemma-3-12b-it;
OpenRouter has no gemma-4-12b listing. These are not actual hosting or electricity
costs. There is no evaluator choice of price mapping or precision.

Use a fresh service and quota state for each cold run. Startup warmup remains
enabled as configured. Do not run selftest or any other requests concurrently:
reconciliation requires exclusive engine counters. Retain the full harness output
and fill accuracy, total USD price, p50/p95 seconds and peak GPU memory in
DISCLOSURE.md. Usage comes from actual engine input/output processing.
