date_kube_schedule ="2024-04-11"
date_ecmus="2024-04-11"
date_ecmus_no_migration="2024-04-23"
date_random="" //TODO: fill here
date_cloud_first="2024-05-14"

all_scenarios: normal wavy

wavy: wavy_average_latency_boxplots wavy_average_latency_linecharts wavy_edge_utilization_linechart wavy_placement_ratio_linechart


normal: normal_average_latency_boxplots normal_average_latency_linecharts normal_edge_utilization_linechart normal_placement_ratio_linechart

wavy_placement_ratio_linechart:
	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5.json.json \
		--scenario-name wavy_scenario_0.5 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_0.5

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_1.json.json \
		--scenario-name wavy_scenario_0.5_1 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_0.5_1

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_2.json.json \
		--scenario-name wavy_scenario_0.5_2 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_0.5_2

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_3.json.json \
		--scenario-name wavy_scenario_0.5_3 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_0.5_3

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_4.json.json \
		--scenario-name wavy_scenario_0.5_4 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_0.5_4

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0.json.json \
		--scenario-name wavy_scenario_1.0 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_1.0

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_1.json.json \
		--scenario-name wavy_scenario_1.0_1 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_1.0_1

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_2.json.json \
		--scenario-name wavy_scenario_1.0_2 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_1.0_2

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_3.json.json \
		--scenario-name wavy_scenario_1.0_3 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_1.0_3

	python3 main.py \
		--script_name placement_ratio_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_4.json.json \
		--scenario-name wavy_scenario_1.0_4 \
		--save-path ./results/placement_ratio/linechart/wavy_scenario_1.0_4

normal_placement_ratio_linechart:
	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.1.json.json \
		--scenario-name normal_scenario_0.5_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.5_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.5.json.json \
		--scenario-name normal_scenario_0.5_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.5_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.7.json.json \
		--scenario-name normal_scenario_0.5_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.5_0.7

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.1.json.json \
		--scenario-name normal_scenario_0.75_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.75_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.5.json.json \
		--scenario-name normal_scenario_0.75_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.75_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.7.json.json \
		--scenario-name normal_scenario_0.75_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.75_0.7

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.1.json.json \
		--scenario-name normal_scenario_1.0_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.0_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.5.json.json \
		--scenario-name normal_scenario_1.0_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.0_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.7.json.json \
		--scenario-name normal_scenario_1.0_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.0_0.7

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.1.json.json \
		--scenario-name normal_scenario_1.25_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.25_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.5.json.json \
		--scenario-name normal_scenario_1.25_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.25_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.7.json.json \
		--scenario-name normal_scenario_1.25_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.25_0.7

wavy_edge_utilization_linechart:
	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5.json.json \
		--scenario-name wavy_scenario_0.5 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_0.5

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_1.json.json \
		--scenario-name wavy_scenario_0.5_1 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_0.5_1

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_2.json.json \
		--scenario-name wavy_scenario_0.5_2 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_0.5_2

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_3.json.json \
		--scenario-name wavy_scenario_0.5_3 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_0.5_3

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_4.json.json \
		--scenario-name wavy_scenario_0.5_4 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_0.5_4

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0.json.json \
		--scenario-name wavy_scenario_1.0 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_1.0

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_1.json.json \
		--scenario-name wavy_scenario_1.0_1 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_1.0_1

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_2.json.json \
		--scenario-name wavy_scenario_1.0_2 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_1.0_2

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_3.json.json \
		--scenario-name wavy_scenario_1.0_3 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_1.0_3

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_4.json.json \
		--scenario-name wavy_scenario_1.0_4 \
		--save-path ./results/edge_utilization/linechart/wavy_scenario_1.0_4

wavy_average_latency_linecharts:
	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5.json.json \
		--scenario-name wavy_scenario_0.5 \
		--save-path ./results/average_latency/linechart/wavy_scenario_0.5

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_1.json.json \
		--scenario-name wavy_scenario_0.5_1 \
		--save-path ./results/average_latency/linechart/wavy_scenario_0.5_1

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_2.json.json \
		--scenario-name wavy_scenario_0.5_2 \
		--save-path ./results/average_latency/linechart/wavy_scenario_0.5_2

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_3.json.json \
		--scenario-name wavy_scenario_0.5_3 \
		--save-path ./results/average_latency/linechart/wavy_scenario_0.5_3

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_4.json.json \
		--scenario-name wavy_scenario_0.5_4 \
		--save-path ./results/average_latency/linechart/wavy_scenario_0.5_4

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0.json.json \
		--scenario-name wavy_scenario_1.0 \
		--save-path ./results/average_latency/linechart/wavy_scenario_1.0

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_1.json.json \
		--scenario-name wavy_scenario_1.0_1 \
		--save-path ./results/average_latency/linechart/wavy_scenario_1.0_1

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_2.json.json \
		--scenario-name wavy_scenario_1.0_2 \
		--save-path ./results/average_latency/linechart/wavy_scenario_1.0_2

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_3.json.json \
		--scenario-name wavy_scenario_1.0_3 \
		--save-path ./results/average_latency/linechart/wavy_scenario_1.0_3

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_4.json.json \
		--scenario-name wavy_scenario_1.0_4 \
		--save-path ./results/average_latency/linechart/wavy_scenario_1.0_4

normal_average_latency_linecharts:
	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.1.json.json \
		--scenario-name normal_scenario_0.5_0.1 \
		--save-path ./results/average_latency/linechart/normal_scenario_0.5_0.1

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.5.json.json \
		--scenario-name normal_scenario_0.5_0.5 \
		--save-path ./results/average_latency/linechart/normal_scenario_0.5_0.5

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.7.json.json \
		--scenario-name normal_scenario_0.5_0.7 \
		--save-path ./results/average_latency/linechart/normal_scenario_0.5_0.7

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.1.json.json \
		--scenario-name normal_scenario_0.75_0.1 \
		--save-path ./results/average_latency/linechart/normal_scenario_0.75_0.1

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.5.json.json \
		--scenario-name normal_scenario_0.75_0.5 \
		--save-path ./results/average_latency/linechart/normal_scenario_0.75_0.5

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.7.json.json \
		--scenario-name normal_scenario_0.75_0.7 \
		--save-path ./results/average_latency/linechart/normal_scenario_0.75_0.7

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.1.json.json \
		--scenario-name normal_scenario_1.0_0.1 \
		--save-path ./results/average_latency/linechart/normal_scenario_1.0_0.1

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.5.json.json \
		--scenario-name normal_scenario_1.0_0.5 \
		--save-path ./results/average_latency/linechart/normal_scenario_1.0_0.5

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.7.json.json \
		--scenario-name normal_scenario_1.0_0.7 \
		--save-path ./results/average_latency/linechart/normal_scenario_1.0_0.7

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.1.json.json \
		--scenario-name normal_scenario_1.25_0.1 \
		--save-path ./results/average_latency/linechart/normal_scenario_1.25_0.1

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.5.json.json \
		--scenario-name normal_scenario_1.25_0.5 \
		--save-path ./results/average_latency/linechart/normal_scenario_1.25_0.5

	python3 main.py \
		--script_name average_latency_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.7.json.json \
		--scenario-name normal_scenario_1.25_0.7 \
		--save-path ./results/average_latency/linechart/normal_scenario_1.25_0.7

wavy_average_latency_boxplots:
	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5.json.json \
		--scenario-name wavy_scenario_0.5 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_1.json.json \
		--scenario-name wavy_scenario_0.5_1 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_0.5_1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_2.json.json \
		--scenario-name wavy_scenario_0.5_2 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_0.5_2

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_3.json.json \
		--scenario-name wavy_scenario_0.5_3 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_0.5_3

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_0.5_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_0.5_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_0.5_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_0.5_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_0.5_4.json.json \
		--scenario-name wavy_scenario_0.5_4 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_0.5_4

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0.json.json \
		--scenario-name wavy_scenario_1.0 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_1.0

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_1.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_1.json.json \
		--scenario-name wavy_scenario_1.0_1 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_1.0_1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_2.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_2.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_2.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_2.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_2.json.json \
		--scenario-name wavy_scenario_1.0_2 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_1.0_2

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_3.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_3.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_3.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_3.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_3.json.json \
		--scenario-name wavy_scenario_1.0_3 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_1.0_3

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/wavy_scenario_1.0_4.json.json \
			reports/kube-schedule/$(date_kube_schedule)/wavy_scenario_1.0_4.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/wavy_scenario_1.0_4.json.json \
			reports/random-scheduler/$(date_random)/wavy_scenario_1.0_4.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/wavy_scenario_1.0_4.json.json \
		--scenario-name wavy_scenario_1.0_4 \
		--save-path ./results/average_latency/boxplot/wavy_scenario_1.0_4

normal_average_latency_boxplots:
	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.1.json.json \
		--scenario-name normal_scenario_0.5_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.5_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.5.json.json \
		--scenario-name normal_scenario_0.5_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.5_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.7.json.json \
		--scenario-name normal_scenario_0.5_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.5_0.7

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.1.json.json \
		--scenario-name normal_scenario_0.75_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.75_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.5.json.json \
		--scenario-name normal_scenario_0.75_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.75_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.7.json.json \
		--scenario-name normal_scenario_0.75_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_0.75_0.7

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.1.json.json \
		--scenario-name normal_scenario_1.0_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.0_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.5.json.json \
		--scenario-name normal_scenario_1.0_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.0_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.7.json.json \
		--scenario-name normal_scenario_1.0_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.0_0.7

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.1.json.json \
		--scenario-name normal_scenario_1.25_0.1 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.25_0.1

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.5.json.json \
		--scenario-name normal_scenario_1.25_0.5 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.25_0.5

	python3 main.py \
		--script_name average_latency_boxplot \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.7.json.json \
		--scenario-name normal_scenario_1.25_0.7 \
		--save-path ./results/average_latency/boxplot/normal_scenario_1.25_0.7

normal_edge_utilization_linechart:
	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.1.json.json \
		--scenario-name normal_scenario_0.5_0.1 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_0.5_0.1

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.5.json.json \
		--scenario-name normal_scenario_0.5_0.5 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_0.5_0.5

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.5_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.5_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.5_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.5_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.5_0.7.json.json \
		--scenario-name normal_scenario_0.5_0.7 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_0.5_0.7

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.1.json.json \
		--scenario-name normal_scenario_0.75_0.1 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_0.75_0.1

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.5.json.json \
		--scenario-name normal_scenario_0.75_0.5 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_0.75_0.5

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_0.75_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_0.75_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_0.75_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_0.75_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_0.75_0.7.json.json \
		--scenario-name normal_scenario_0.75_0.7 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_0.75_0.7

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.1.json.json \
		--scenario-name normal_scenario_1.0_0.1 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_1.0_0.1

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.5.json.json \
		--scenario-name normal_scenario_1.0_0.5 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_1.0_0.5

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.0_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.0_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.0_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.0_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.0_0.7.json.json \
		--scenario-name normal_scenario_1.0_0.7 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_1.0_0.7

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.1.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.1.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.1.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.1.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.1.json.json \
		--scenario-name normal_scenario_1.25_0.1 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_1.25_0.1

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.5.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.5.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.5.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.5.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.5.json.json \
		--scenario-name normal_scenario_1.25_0.5 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_1.25_0.5

	python3 main.py \
		--script_name edge_utilization_linechart \
		--config_path config.json \
		--history_paths \
			reports/ecmus/$(date_ecmus)/normal_scenario_1.25_0.7.json.json \
			reports/kube-schedule/$(date_kube_schedule)/normal_scenario_1.25_0.7.json.json \
			reports/ecmus-no-migration/$(date_ecmus_no_migration)/normal_scenario_1.25_0.7.json.json \
			reports/random-scheduler/$(date_random)/normal_scenario_1.25_0.7.json.json \
			reports/cloud-first-scheduler/$(date_cloud_first)/normal_scenario_1.25_0.7.json.json \
		--scenario-name normal_scenario_1.25_0.7 \
		--save-path ./results/edge_utilization/linechart/normal_scenario_1.25_0.7
