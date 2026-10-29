# InterviewTrack AI — common tasks
.PHONY: help setup demo test eval app auth clean

help:           ## show available targets
	@grep -hE '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /  —  /'

setup:          ## one-command setup (venv, deps, tests)
	./scripts/setup.sh

app:            ## launch the Streamlit UI (Demo Mode needs no credentials)
	streamlit run app.py

demo:           ## headless synthetic reconstruction + ground-truth checks
	python scripts/demo.py

eval:           ## cross-inbox generalization report (3 mailbox profiles)
	python scripts/evaluate_cross_inbox.py

test:           ## full test suite (offline, no model download)
	python -m pytest -q

auth:           ## authorize Gmail Mode (read-only, one-time)
	python -m interviewtrack.auth

clean:          ## remove local data and caches (keeps tokens)
	rm -rf data/*.db data/*.db-wal data/*.db-shm .pytest_cache __pycache__ \
	       interviewtrack/__pycache__ tests/__pycache__
