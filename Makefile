soft: 
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_soft.json reports/kube-schedule/2023-12-31/status_soft.json 

mid: 
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_mid.json reports/kube-schedule/2023-12-31/status_mid.json 

hard: 
	python3 main.py --script_name check_equality --config_path config.json --history_paths reports/ecmus/2023-12-31/status_hard.json reports/kube-schedule/2023-12-31/status_hard.json
