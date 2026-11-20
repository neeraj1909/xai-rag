#!/usr/bin/env bash
# Uses only newly created, uniquely named local resources. The provider ledger
# includes complete prompts, so use a fictional fixture and protect the output.
set -euo pipefail
umask 077

runner="${XAI_EVAL_RUNNER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
project="${XAI_RAG_PROJECT_DIR:-$(cd "$runner/../.." && pwd)}"
python="${XAI_EVAL_PYTHON:-$project/.venv/bin/python}"
run_root="$(realpath -m "${XAI_EVAL_RUN_ROOT:?set a new unique output directory}")"
chroma_port="${XAI_EVAL_CHROMA_PORT:?set leased Chroma port}"
es_port="${XAI_EVAL_ES_PORT:?set leased Elasticsearch port}"
api_port="${XAI_EVAL_API_PORT:?set leased API port}"
run_name="${XAI_EVAL_RUN_NAME:?set a unique owned docker name prefix}"
test "${XAI_EVAL_MAX_PROVIDER_CALLS:?set shared provider call cap}" -ge 1
export XAI_EVAL_MODEL_SEED="${XAI_EVAL_MODEL_SEED:-20261120}"
export XAI_EVAL_FIXTURE="${XAI_EVAL_FIXTURE:-$project/evals/results/2026-11-20-synthetic-http/holdout-cases.json}"
runs="${XAI_EVAL_RUNS:-3}"
test "$runs" -ge 1
export XAI_EVAL_RUN_FILE="$run_root/current-run"
test ! -e "$run_root"
mkdir -m 700 "$run_root"
export XAI_EVAL_BUDGET_FILE="${XAI_EVAL_BUDGET_FILE:-$run_root/provider-attempts.jsonl}"
export XAI_RAG_CHROMA_HOST=127.0.0.1 XAI_RAG_CHROMA_PORT="$chroma_port"
export XAI_RAG_ELASTICSEARCH_URL="http://127.0.0.1:$es_port"
export XAI_RAG_CHROMA_COLLECTION="xai_http_eval" XAI_RAG_ELASTICSEARCH_INDEX="xai_http_eval"
# Match the recorded run and prevent inherited settings from touching another journal.
export XAI_RAG_INGESTION_JOURNAL_ROOT=""
export XAI_RAG_LLM_MODEL="${XAI_RAG_LLM_MODEL:-gpt-6-luna}" XAI_RAG_RAGAS_LLM_MODEL="${XAI_RAG_RAGAS_LLM_MODEL:-gpt-6-luna}"
export XAI_RAG_GROUNDING_JUDGE_MODEL="${XAI_RAG_GROUNDING_JUDGE_MODEL:-gpt-6-luna}" XAI_RAG_GROUNDING_JUDGE_PROVIDER=openai
export XAI_RAG_LLM_BASE_URL=https://api.openai.com/v1
export XAI_RAG_GROUNDING_JUDGE_BASE_URL=https://api.openai.com/v1
export XAI_RAG_OTEL_ENDPOINT=
export XAI_RAG_RATE_LIMIT_PER_MINUTE=1000 XAI_RAG_QUERY_TIMEOUT_SECONDS=300
export TOKENIZERS_PARALLELISM=false ANONYMIZED_TELEMETRY=false
# Some environments require their proxy. Override only for this process when
# direct provider access is known to work and the inherited proxy is broken.
if [ "${XAI_EVAL_DISABLE_PROXY:-0}" = 1 ]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
fi
cd "$project"
docker_root="$(docker info --format '{{.DockerRootDir}}')"

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
    free_disk=$(df -Pk "$docker_root" | awk 'NR==2 {print $4}')
    free_tmp=$(df -Pk "$run_root" | awk 'NR==2 {print $4}')
    free_memory=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    free_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    total_swap=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
    if [ "$free_disk" -lt 10485760 ] || [ "$free_tmp" -lt 3145728 ] || [ "$free_memory" -lt 3145728 ] || \
       { [ "$total_swap" -gt 0 ] && [ "$free_swap" -lt 1048576 ]; } || [ $((SECONDS - started)) -ge 14400 ]; then
      printf '%s\n' 'Resource/time watchdog stopped the owned synthetic evaluation.' >"$run_root/watchdog-stop.txt"
      for child in $(pgrep -P "$main_pid" || true); do
        if [ "$child" != "$BASHPID" ]; then kill -TERM "$child" 2>/dev/null || true; fi
      done
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
docker inspect --format '{{.Image}}' "${run_name}-chroma" > "$run_root/chroma-image-id.txt"
docker inspect --format '{{.Image}}' "${run_name}-es" > "$run_root/es-image-id.txt"

for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$chroma_port/api/v2/heartbeat" >/dev/null && \
     curl -fsS "http://127.0.0.1:$es_port/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null; then break; fi
  if [ "$attempt" = 120 ]; then echo 'Disposable store readiness timeout' >&2; exit 1; fi
  sleep 1
done

export XAI_EVAL_DIR="$run_root/holdout"
"$python" "$runner/prepare.py" --fixture "$XAI_EVAL_FIXTURE" --root "$XAI_EVAL_DIR"
export XAI_RAG_INGEST_ROOT="$XAI_EVAL_DIR/documents" XAI_RAG_AUTH_POLICY_FILE="$XAI_EVAL_DIR/policy.json"
export XAI_EVAL_CAPTURE_PATH="$XAI_EVAL_DIR/capture.jsonl"
"$python" "$runner/manifest.py" --output "$run_root/manifest.json" --fixture "$XAI_EVAL_FIXTURE" --runner-dir "$runner"
"$python" "$runner/capture_server.py" >"$run_root/holdout-api.log" 2>&1 & api_pid=$!
for attempt in $(seq 1 120); do
  if kill -0 "$api_pid" 2>/dev/null && "$python" "$runner/owns_port.py" "$api_pid" "$api_port" && \
     curl -fsS "http://127.0.0.1:$api_port/ready" >/dev/null; then break; fi
  if ! kill -0 "$api_pid" 2>/dev/null || [ "$attempt" = 120 ]; then echo 'Evaluation API startup failure' >&2; exit 1; fi
  sleep 1
done
"$python" "$runner/repeat_http.py" --port "$api_port" --root "$XAI_EVAL_DIR" --fixture "$XAI_EVAL_FIXTURE" --output-dir "$run_root" --seed 20261120 --mode ingest
for run in $(seq 1 "$runs"); do
  printf '%s' "$run" > "$XAI_EVAL_RUN_FILE"
  "$python" "$runner/index_cli.py" --fixture "$XAI_EVAL_FIXTURE" --output "$run_root/index-before-$run.json"
  "$python" "$runner/repeat_http.py" --port "$api_port" --root "$XAI_EVAL_DIR" --fixture "$XAI_EVAL_FIXTURE" --output-dir "$run_root" --seed 20261120 --mode query --run "$run"
  "$python" "$runner/index_cli.py" --fixture "$XAI_EVAL_FIXTURE" --output "$run_root/index-after-$run.json"
  "$python" "$runner/compare_indexes.py" "$run_root/index-before-$run.json" "$run_root/index-after-$run.json" "$run_root/index-before-1.json"
  printf -v label '%02d' "$run"
  "$python" "$runner/ragas_score.py" --input "$run_root/run-$label-http.json" --output "$run_root/run-$label-ragas.json" --seed 20261120 --preserve-order
 done
"$python" "$runner/summarize_repeats.py" --fixture "$XAI_EVAL_FIXTURE" --directory "$run_root" --runs "$runs" --model-seed "$XAI_EVAL_MODEL_SEED" > "$run_root/summary.json"
echo "Synthetic evaluation complete; evidence: $run_root"
