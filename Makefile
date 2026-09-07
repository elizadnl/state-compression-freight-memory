.PHONY: prepare analysis checks simulations clean

prepare:
	python src/prepare_public_bdi.py

analysis:
	python src/analyze_public_bdi.py

checks:
	python src/identification_checks.py

simulations:
	python src/analyze_public_bdi.py --simulations --b-week 1000 --b-roll 500

clean:
	rm -f data/public_bdi_2009_2025.csv
