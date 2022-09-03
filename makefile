
PYTHON=python3.8
CONDA=conda

run:
	eval "$$($(CONDA) shell.bash hook)" && conda activate komma && $(PYTHON) komma/model.py config/model.toml

install:
	$(CONDA) env create -f conda.yml
