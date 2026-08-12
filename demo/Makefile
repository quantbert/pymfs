.PHONY: data demo clean

data:
	uv run --env-file .env --group generation python run.py

demo:
	uv run --env-file .env --extra pandas python demo.py

clean:
	rm -rf data feature_cache