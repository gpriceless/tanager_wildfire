.PHONY: install test lint notebooks figures clean all

install:
	pip install -e ".[dev,notebook]"

test:
	pytest

lint:
	ruff check .

notebooks:
	jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb

figures:
	mkdir -p figures
	cp outputs/notebook02_severity_summary.png figures/severity_map.png
	cp outputs/notebook04_trajectories.png figures/temporal_trajectory.png
	cp outputs/notebook03_water_indices.png figures/water_indices.png
	cp outputs/notebook05_information_loss_curve.png figures/sensor_comparison.png
	cp outputs/notebook04_severity_stratified.png figures/severity_stratified.png
	cp outputs/notebook05_spectral_response.png figures/spectral_response.png

clean:
	rm -rf figures
	find outputs -type f \( -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) -delete
	find notebooks -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} +

all: install test lint
