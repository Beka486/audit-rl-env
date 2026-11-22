PY ?= python3.11
IMAGE ?= auditenv:1.0.0

.PHONY: check test generate trajectories report demo docker-build docker-check

check:            ## the single official verification command
	$(PY) -m auditenv check

test:
	$(PY) -m unittest discover -s tests -t . -v

generate:         ## re-export the 10 train + 10 dev episodes
	$(PY) -m auditenv generate

trajectories:
	$(PY) -m auditenv trajectories

report:
	$(PY) -m auditenv report

demo:
	$(PY) -m auditenv demo --split dev --index 5

docker-build:
	docker build -t $(IMAGE) .

docker-check: docker-build   ## offline, 2 CPU, 4 GB, as in the official limits
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
