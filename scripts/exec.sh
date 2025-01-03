#!/bin/sh

if [ -d "results" ]; then
  echo "Results directory already exists. Remove it first? (y/n)"
  read answer
  if [ "$answer" = "y" ]; then
    rm -rf results
    mkdir results
  else
    echo "Operation cancelled"
    exit 1
  fi
else
  mkdir results
fi

EXTRACTORS=$(cat $1)
SCENARIOS=$(cat $2)
SCHEDULERS=$(cat $3)

echo "Extractors: $EXTRACTORS"
echo "Scenarios: $SCENARIOS"
echo "Schedulers: $SCHEDULERS"

for extractor in $EXTRACTORS; do
  mkdir results/$extractor
  for scenario in $SCENARIOS; do
    mkdir results/$extractor/$scenario
    cmd="python3 main.py --extractor_name $extractor --config_path config.json --save_path results/$extractor/$scenario --reports_dir reports --schedulers_names $SCHEDULERS --scenario_name $scenario"
    echo "Running $cmd"
    $cmd
    
    if [ $? -ne 0 ]; then
      echo "Error running $extractor with $scenario"
      exit 1
    fi
  done
done
