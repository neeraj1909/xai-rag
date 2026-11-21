#!/usr/bin/env bash
# Only run after the coordinator grants a lease for these loopback ports.
set -euo pipefail
umask 077

project=/home/neeraj/Documents/xai-rag
python="$project/.venv/bin/python"
runner=/tmp/xai-http-evaluation
run_root="${XAI_EVAL_RUN_ROOT:?set a new unique output directory}"
chroma_port="${XAI_EVAL_CHROMA_PORT:?set leased Chroma port}"
es_port="${XAI_EVAL_ES_PORT:?set leased Elasticsearch port}"
api_port="${XAI_EVAL_API_PORT:?set leased API port}"
run_name="${XAI_EVAL_RUN_NAME:?set a unique owned docker name prefix}"
test "${XAI_EVAL_MAX_PROVIDER_CALLS:?set shared provider call cap}" = 350
test ! -e "$run_root"
mkdir -m 700 "$run_root"
export XAI_EVAL_BUDGET_FILE="${XAI_EVAL_BUDGET_FILE:-$run_root/provider-attempts.jsonl}"
export XAI_RAG_CHROMA_HOST=127.0.0.1 XAI_RAG_CHROMA_PORT="$chroma_port"
export XAI_RAG_ELASTICSEARCH_URL="http://127.0.0.1:$es_port"
export XAI_RAG_CHROMA_COLLECTION="xai_http_eval" XAI_RAG_ELASTICSEARCH_INDEX="xai_http_eval"
export XAI_RAG_LLM_MODEL=gpt-6-luna XAI_RAG_RAGAS_LLM_MODEL=gpt-6-luna
export XAI_RAG_GROUNDING_JUDGE_MODEL=gpt-6-luna XAI_RAG_GROUNDING_JUDGE_PROVIDER=openai
export XAI_RAG_LLM_BASE_URL=https://api.openai.com/v1
export XAI_RAG_GROUNDING_JUDGE_BASE_URL=https://api.openai.com/v1
export XAI_RAG_OTEL_ENDPOINT=
export XAI_RAG_RATE_LIMIT_PER_MINUTE=1000 XAI_RAG_QUERY_TIMEOUT_SECONDS=300
export TOKENIZERS_PARALLELISM=false ANONYMIZED_TELEMETRY=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
cd "$project"

api_pid=
watchdog_pid=
chroma_created=0
es_created=0
cleanup() {
  if [ -n "$watchdog_pid" ]; then kill "$watchdog_pid" 2>/dev/null || true; wait "$watchdog_pid" 2>/dev/null || true; fi
  if [ -n "$api_pid" ]; then kill "$api_pid" 2>/dev/null || true; wait "$api_pid" 2>/dev/null || true; fi
  if [ "$chroma_created" = 1 ]; then docker rm -f "${run_name}-chroma" >/dev/null 2>&1 || true; fi
  if [ "$es_created" = 1 ]; then docker rm -f "${run_name}-es" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
main_pid=$$
(
  started=$SECONDS
  while kill -0 "$main_pid" 2>/dev/null; do
    free_disk=$(df -Pk /var/lib/docker | awk 'NR==2 {print $4}')
    free_tmp=$(df -Pk "$run_root" | awk 'NR==2 {print $4}')
    free_memory=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    free_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    if [ "$free_disk" -lt 10485760 ] || [ "$free_tmp" -lt 3145728 ] || [ "$free_memory" -lt 2621440 ] || \
       [ "$free_swap" -lt 1048576 ] || [ $((SECONDS - started)) -ge 7200 ]; then
      printf '%s\n' 'Resource/time watchdog stopped the owned synthetic evaluation.' >"$run_root/watchdog-stop.txt"
      kill -TERM "$main_pid" 2>/dev/null || true
      exit
    fi
    sleep 10
  done
) & watchdog_pid=$!

docker run -d --rm --name "${run_name}-chroma" -p "127.0.0.1:${chroma_port}:8000" \
  -e ANONYMIZED_TELEMETRY=false chromadb/chroma:1.5.9 >/dev/null
chroma_created=1
docker run -d --rm --name "${run_name}-es" -p "127.0.0.1:${es_port}:9200" \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e cluster.routing.allocation.disk.threshold_enabled=false \
  -e 'ES_JAVA_OPTS=-Xms512m -Xmx512m' \
  docker.elastic.co/elasticsearch/elasticsearch:8.17.0 >/dev/null
es_created=1

for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$chroma_port/api/v2/heartbeat" >/dev/null && \
     curl -fsS "http://127.0.0.1:$es_port/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null; then break; fi
  if [ "$attempt" = 120 ]; then echo 'Disposable store readiness timeout' >&2; exit 1; fi
  sleep 1
done

export XAI_EVAL_DIR="$run_root/holdout"
"$python" "$runner/prepare.py"
export XAI_RAG_INGEST_ROOT="$XAI_EVAL_DIR/documents" XAI_RAG_AUTH_POLICY_FILE="$XAI_EVAL_DIR/policy.json"
export XAI_EVAL_CAPTURE_PATH="$XAI_EVAL_DIR/capture.jsonl"
"$python" "$runner/manifest.py" --output "$run_root/manifest.json"
"$python" "$runner/capture_server.py" >"$run_root/holdout-api.log" 2>&1 & api_pid=$!
for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$api_port/ready" >/dev/null; then break; fi
  if ! kill -0 "$api_pid" 2>/dev/null || [ "$attempt" = 120 ]; then echo 'Evaluation API startup failure' >&2; exit 1; fi
  sleep 1
done
mapfile -t cases < <("$python" -c "import json; print('\n'.join(x['id'] for x in json.load(open('/tmp/xai-e2e-harness/holdout-cases.json'))['queries']))")
"$python" "$runner/run_http.py" --port "$api_port" --root "$XAI_EVAL_DIR" \
  --output "$run_root/holdout-http.json" --cases "${cases[@]}"
"$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(bool(x.get("halt_reason")))' "$run_root/holdout-http.json"
kill "$api_pid"; wait "$api_pid" || true; api_pid=

export XAI_EVAL_DIR="$run_root/security"
"$python" "$runner/prepare_security.py"
export XAI_RAG_INGEST_ROOT="$XAI_EVAL_DIR/documents" XAI_RAG_AUTH_POLICY_FILE="$XAI_EVAL_DIR/policy.json"
export XAI_EVAL_CAPTURE_PATH="$XAI_EVAL_DIR/capture.jsonl"
"$python" "$runner/capture_server.py" >"$run_root/security-api.log" 2>&1 & api_pid=$!
for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$api_port/ready" >/dev/null; then break; fi
  if ! kill -0 "$api_pid" 2>/dev/null || [ "$attempt" = 120 ]; then echo 'Security API startup failure' >&2; exit 1; fi
  sleep 1
done
"$python" "$runner/run_security.py" --port "$api_port" --root "$XAI_EVAL_DIR" \
  --output "$run_root/security-http.json"
"$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(bool(x.get("halt_reason")))' "$run_root/security-http.json"
kill "$api_pid"; wait "$api_pid" || true; api_pid=

"$python" "$runner/judge_security.py" --input "$run_root/security-http.json" \
  --output "$run_root/security-judge.json" --model gpt-6-luna --max-calls 28
"$python" "$runner/ragas_score.py" --input "$run_root/holdout-http.json" \
  --output "$run_root/holdout-ragas.json"
"$python" "$runner/to_dataset.py" --input "$run_root/holdout-http.json" \
  --fixture /tmp/xai-e2e-harness/holdout-cases.json --ragas "$run_root/holdout-ragas.json" \
  --output "$run_root/holdout-evaluator.jsonl" --config-revision "$(date -u +%Y-%m-%d)-http"
"$python" "$runner/summarize.py" --input "$run_root/holdout-http.json" --ragas "$run_root/holdout-ragas.json" \
  --budget "$XAI_EVAL_BUDGET_FILE" >"$run_root/holdout-summary.json"
"$python" "$runner/summarize.py" --input "$run_root/security-http.json" \
  --security-judge "$run_root/security-judge.json" --budget "$XAI_EVAL_BUDGET_FILE" >"$run_root/security-summary.json"
echo "Synthetic evaluation complete; evidence: $run_root"
