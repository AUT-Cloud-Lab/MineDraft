check_equality_soft:
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_soft.json reports/kube-schedule/2023-12-31/status_soft.json --save-path ./results/check_equality/soft.txt

check_equality_mid:
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_mid.json reports/kube-schedule/2023-12-31/status_mid.json --save-path ./results/check_equality/mid.txt

check_equality_hard:
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_hard.json reports/kube-schedule/2023-12-31/status_hard.json --save-path ./results/check_equality/hard.txt

