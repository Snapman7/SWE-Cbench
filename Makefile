run:
	python3 pipeline/run_pipeline.py

benchmark:
	python3 benchmark/build_benchmark.py

inference:
	python3 inference/run_model.py

eval:
	python3 evaluation/evaluate.py

docker-eval:
	docker-compose up --build eval

