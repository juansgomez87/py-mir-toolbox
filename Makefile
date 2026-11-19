# Build and deploy the interactive site (site/) with the Sphinx docs under /docs/.
#   make site     build everything into _site/
#   make deploy   copy _site/ to the CCRMA web space (needs SSH access)
#   make parity-report   compare with the local MIRtoolbox fixtures and save the
#                        results shown on the docs Parity report page

SITE_DIR     := _site
CCRMA_HOST   ?= ccrma-gate.stanford.edu
CCRMA_USER   ?= jsgomezc
CCRMA_PATH   ?= Library/Web/py-mir-toolbox
FIXTURES_DIR ?= validation/fixtures
PARITY_JSON  := docs/parity/results.json

.PHONY: docs site deploy clean parity-report

docs:
	uv run --group docs sphinx-build -W -b html docs docs/_build/html

site: docs
	rm -rf $(SITE_DIR)
	cp -R site $(SITE_DIR)
	cp -R docs/_build/html $(SITE_DIR)/docs

deploy: site
	rsync -avz --delete $(SITE_DIR)/ $(CCRMA_USER)@$(CCRMA_HOST):$(CCRMA_PATH)/

# The parity run exits non-zero while a case fails (e.g. the accepted
# filterbank_2channels difference); the report is written either way.
parity-report:
	-MIRPY_FIXTURES=$(FIXTURES_DIR) MIRPY_PARITY_REPORT=$(PARITY_JSON) \
		uv run --group test pytest tests/parity -q
	@test -f $(PARITY_JSON) && echo "Wrote $(PARITY_JSON)"

clean:
	rm -rf $(SITE_DIR) docs/_build docs/generated docs/jupyter_execute
