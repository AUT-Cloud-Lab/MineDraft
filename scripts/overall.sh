#!/bin/sh

EXTRACTORS="all_data_tables pod_count_linechart average_latency_linechart average_latency_boxplot edge_utilization_linechart fragmentation_data"
SCENARIOS="normal_scenario_1.1_0.4 normal_scenario_1.2_0.4 normal_scenario_1.3_0.4 normal_scenario_1.4_0.4 normal_scenario_1.5_0.1 normal_scenario_1.5_0.2 normal_scenario_1.5_0.3 normal_scenario_1.5_0.4 normal_scenario_1.5_0.5 normal_scenario_1.6_0.4"
SCHEDULERS="kube-schedule ecmus-qos-aware smallest-edge-first-scheduler biggest-edge-first-scheduler random-scheduler cloud-first-scheduler"

if [ -d "tmp" ]; then
  rm -rf tmp
fi
mkdir tmp

echo $EXTRACTORS > tmp/extractors
echo $SCENARIOS > tmp/scenarios
echo $SCHEDULERS > tmp/schedulers

scripts/exec.sh tmp/extractors tmp/scenarios tmp/schedulers

rm -rf tmp
