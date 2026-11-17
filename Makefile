# Build and deploy the interactive site (site/) with the Sphinx docs under /docs/.
#   make site     build everything into _site/
#   make deploy   copy _site/ to the CCRMA web space (needs SSH access)

SITE_DIR     := _site
CCRMA_HOST   ?= ccrma-gate.stanford.edu
CCRMA_USER   ?= jsgomezc
CCRMA_PATH   ?= Library/Web/py-mir-toolbox

.PHONY: docs site deploy clean

docs:
	uv run --group docs sphinx-build -W -b html docs docs/_build/html

site: docs
	rm -rf $(SITE_DIR)
	cp -R site $(SITE_DIR)
	cp -R docs/_build/html $(SITE_DIR)/docs

deploy: site
	rsync -avz --delete $(SITE_DIR)/ $(CCRMA_USER)@$(CCRMA_HOST):$(CCRMA_PATH)/

clean:
	rm -rf $(SITE_DIR) docs/_build docs/generated docs/jupyter_execute
