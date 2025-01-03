#!/bin/sh

EXTRACTORS="pod_count_linechart average_latency_linechart average_latency_boxplot edge_utilization_linechart"
SCENARIOS="normal_scenario_1.3_0.4 normal_scenario_1.4_0.4"
SCHEDULERS="kube-schedule ecmus-qos-aware ecmus-all-half ecmus-c-gt-a ecmus-only-d"

if [ -d "tmp" ]; then
  rm -rf tmp
fi
mkdir tmp

echo $EXTRACTORS > tmp/extractors
echo $SCENARIOS > tmp/scenarios
echo $SCHEDULERS > tmp/schedulers

scripts/exec.sh tmp/extractors tmp/scenarios tmp/schedulers

rm -rf tmp
