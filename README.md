# InterviewTrack AI

Reconstruct interview timelines from email using deterministic rules, matching evidence and optional local embeddings. The dashboard shows Active and Rejected; an offer is an Active application at the Offer stage.

## Run for review

Install and start Docker Desktop. From this folder run:

```sh
docker compose up --build
```

Open http://localhost:8501, select **Fictional inbox — screening through offer**, and click **Start Demo Mode**. No Gmail authorization is needed. The first build needs internet to install dependencies. The default demo uses deterministic text features, with no neural model download.

Stop with Ctrl+C and `docker compose down`. Runtime data persists in a Docker volume.

Optional neural mode:

```sh
docker compose -f compose.yaml -f compose.neural.yaml up --build
```

This installs PyTorch and downloads the embedding model on first use.

## Fictional SQLite database

`demo/fictional-inbox.db` contains a fresh database generated through the production pipeline. It preserves the application's schema, not the original private database contents. The default demo selection reconstructs the same fictional messages into the runtime database. The bundled database can also be inspected directly with a SQLite viewer.

Avery Quinn and the Gmail-formatted address are fictional display values, not a created or authorized account. Employer addresses use reserved .example domains. Names, companies, dates, IDs and text are invented. Broad screening, clarification and cross-thread patterns informed the examples; offer and rejection cases are additional synthetic coverage, not claims about a real candidate's outcomes. No original message text, attachments, URLs, tokens or private database pages were copied.

There are 31 messages: six interview journeys plus advertisement and receipt examples. The receipt remains internal context but does not create an interview card. The scenarios include separate roles at one employer, follow-ups, an offer and rejection.

Regenerate the bundled database with `python scripts/build_demo_db.py` only after moving the existing fictional database aside. The script refuses to overwrite it and never reads Gmail.

## Local Python alternative

Use Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
EMBEDDINGS_ENABLED=false python -m streamlit run app.py
```

For real Gmail, enable Gmail API in your Google Cloud project, configure your OAuth consent/test user, create a Desktop OAuth client, and save its JSON privately as `data/client_secret.json`. Run `python -m interviewtrack.auth --help` for the authorization command options. Real Gmail requires your own authorization. The Docker reviewer demo does not mount credentials; use local Python for the desktop browser authorization flow.

## Validation and limits

Run `python -m pytest -q` or `docker compose run --rm app python -m pytest -q`. Neural tests may skip with core dependencies. English recruiting rules can miss unfamiliar wording; uncertain matches need review. Passing fictional cases is not a guarantee for every mailbox. Follow-up messages update activity and do not prove an interview was completed. The Docker configuration was validated; an image build and container runtime test were not performed because the local Docker engine was unavailable.

## Repository

Source repository: https://github.com/madhumitha-gv/InterviewTrackAI

For changes after cloning the repository:

```sh
git add .
git status --short
git commit -m "Describe your changes"
git push origin main
```

Private runtime data and credentials are ignored. Only the explicitly generated fictional database is allowed into Git.

See `demo/SCENARIOS.md` for the fictional interview journeys.
