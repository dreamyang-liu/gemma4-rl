#!/usr/bin/env bash
# Stop one experiment: its agent server first (so that no new trajectory starts), the Miles container, the proxy,
# the harness containers, and the task sandboxes that the killed harnesses leave behind.
#   ./stop_experiment.sh <name>
# swegemma's sandboxes carry no label of the experiment, so the orphans are the rebench/* containers created after
# runs/<name>/started_at. They are removed only if no other harness (g4-* container) is running, since a concurrent
# run of another experiment would have sandboxes from the same window; otherwise the script only warns.
NAME=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ps -eo pid,args | awk -v p="--run-prefix ${NAME}_" '$2 ~ /python/ && $3 ~ /agent_server/ && index($0, p) {print $1}' | xargs -r kill
sleep 2
docker rm -f g4rl-miles-$NAME g4rl-proxy-$NAME >/dev/null 2>&1
docker ps --format '{{.Names}}' | grep "^g4-${NAME}_" | xargs -r docker rm -f >/dev/null
if [ -f "$HERE/runs/$NAME/started_at" ]; then
  T0=$(cat "$HERE/runs/$NAME/started_at")
  OTHERS=$(docker ps --format '{{.Names}}' | grep '^g4-' | grep -v -E 'proxy|vserve' | wc -l)
  ORPHANS=$(docker ps --format '{{.ID}}\t{{.Image}}\t{{.CreatedAt}}' | awk -F'\t' -v t="$T0" '$2 ~ /^rebench\// && $3 >= t {print $1}')
  if [ -n "$ORPHANS" ] && [ "$OTHERS" -eq 0 ]; then
    echo "$ORPHANS" | xargs docker rm -f >/dev/null
    echo "removed $(echo "$ORPHANS" | wc -l) task sandboxes created after $T0"
  elif [ -n "$ORPHANS" ]; then
    echo "WARNING: $(echo "$ORPHANS" | wc -l) rebench sandboxes created after $T0 are left, because $OTHERS other harness containers run"
  fi
fi
[ -d "$HERE/runs/$NAME" ] && docker run --rm -v "$HERE/runs/$NAME":/o alpine:3 chown -R "$(id -u):$(id -g)" /o 2>/dev/null
echo "stopped $NAME"
