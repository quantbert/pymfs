DATA_DIRECTORY ?= data
START_DATE ?= 2010-01-01
END_DATE ?= 2025-01-01
TICKER_START ?= 0
TICKER_COUNT ?= 100
SEED ?= 42
OVERWRITE ?= false
DRY_RUN ?= false

OVERWRITE_FLAG := $(if $(filter true 1 yes,$(OVERWRITE)),--overwrite,)
DRY_RUN_FLAG := $(if $(filter true 1 yes,$(DRY_RUN)),--dry-run,)

.PHONY: data generate upload demo clean

data: generate upload

generate:
	uv run --group generation python gendata.py \
		--output "$(DATA_DIRECTORY)/ohlcv" \
		--start "$(START_DATE)" \
		--end "$(END_DATE)" \
		--ticker-start "$(TICKER_START)" \
		--ticker-count "$(TICKER_COUNT)" \
		--seed "$(SEED)" \
		--generate-sma \
		--sma-output "$(DATA_DIRECTORY)/sma" \
		--symbology-output "$(DATA_DIRECTORY)/symbols/data.parquet" \
		--markets-output "$(DATA_DIRECTORY)/markets/data.parquet" $(OVERWRITE_FLAG)

upload:
	uv run --env-file .env python hfupload.py \
		--data "$(DATA_DIRECTORY)" $(DRY_RUN_FLAG)

demo:
	uv run --env-file .env --extra pandas python demo.py

clean:
	rm -rf "$(DATA_DIRECTORY)" feature_cache