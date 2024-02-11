# -------------------------------------------- check_equality -----------------------------------------------
check_equality_soft:
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_soft.json reports/kube-schedule/2023-12-31/status_soft.json --save-path ./results/check_equality/soft.txt

check_equality_mid:
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_mid.json reports/kube-schedule/2023-12-31/status_mid.json --save-path ./results/check_equality/mid.txt

check_equality_hard:
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_hard.json reports/kube-schedule/2023-12-31/status_hard.json --save-path ./results/check_equality/hard.txt
# -------------------------------------------- end check_equality -----------------------------------------------

# -------------------------------------------- calc_migrations -----------------------------------------------
calc_migrations_ecmus_soft:
	python3 main.py --script_name calc_migrations --config_path config.json --history_paths reports/ecmus/2023-12-31/status_soft.json  --save-path ./results/migrations/ecmus/soft.txt

calc_migrations_ecmus_mid:
	python3 main.py --script_name calc_migrations --config_path config.json --history_paths reports/ecmus/2023-12-31/status_mid.json  --save-path ./results/migrations/ecmus/mid.txt

calc_migrations_ecmus_hard:
	python3 main.py --script_name calc_migrations --config_path config.json --history_paths reports/ecmus/2023-12-31/status_hard.json  --save-path ./results/migrations/ecmus/hard.txt

calc_migrations_kube_soft:
	python3 main.py --script_name calc_migrations --config_path config.json --history_paths  reports/kube-schedule/2023-12-31/status_soft.json --save-path ./results/migrations/kube-schedule/soft.txt

calc_migrations_kube_mid:
	python3 main.py --script_name calc_migrations --config_path config.json --history_paths  reports/kube-schedule/2023-12-31/status_mid.json --save-path ./results/migrations/kube-schedule/mid.txt

calc_migrations_kube_hard:
	python3 main.py --script_name calc_migrations --config_path config.json --history_paths  reports/kube-schedule/2023-12-31/status_hard.json --save-path ./results/migrations/kube-schedule/hard.txt
# -------------------------------------------- end calc_migrations -----------------------------------------------

# -------------------------------------------- average_latency_linechart -----------------------------------------------
average_latency_linechart_ecmus_soft:
	python3 main.py --script_name average_latency_linechart --config_path config.json --history_paths reports/ecmus/2023-12-31/status_soft.json  --save-path ./results/average_latency/ecmus/line-chart/soft.png

average_latency_linechart_ecmus_mid:
	python3 main.py --script_name average_latency_linechart --config_path config.json --history_paths reports/ecmus/2023-12-31/status_mid.json  --save-path ./results/average_latency/ecmus/line-chart/mid.png

average_latency_linechart_ecmus_hard:
	python3 main.py --script_name average_latency_linechart --config_path config.json --history_paths reports/ecmus/2023-12-31/status_hard.json  --save-path ./results/average_latency/ecmus/line-chart/hard.png

average_latency_linechart_kube_soft:
	python3 main.py --script_name average_latency_linechart --config_path config.json --history_paths reports/kube-schedule/2023-12-31/status_soft.json  --save-path ./results/average_latency/kube-schedule/line-chart/soft.png

average_latency_linechart_kube_mid:
	python3 main.py --script_name average_latency_linechart --config_path config.json --history_paths reports/kube-schedule/2023-12-31/status_mid.json  --save-path ./results/average_latency/kube-schedule/line-chart/mid.png

average_latency_linechart_kube_hard:
	python3 main.py --script_name average_latency_linechart --config_path config.json --history_paths reports/kube-schedule/2023-12-31/status_hard.json  --save-path ./results/average_latency/kube-schedule/line-chart/hard.png
# -------------------------------------------- end average_latency_linechart -----------------------------------------------
