# InterviewTrack AI

InterviewTrack AI helps prevent screening and interview emails from getting buried in a busy inbox. It identifies actionable recruiting messages and tracks each company and role from the first invitation through later rounds, assessments, follow-ups, and the final offer or rejection.

I built this because important invitations and next steps can be missed among job alerts, advertisements, and application receipts. One application can also involve recruiters, staffing specialists, an employer, and scheduling services across different threads. The app brings those messages into one place so users can keep track of their interview processes.

The dashboard has two views: **Active** and **Rejected**. An offer appears under Active with the stage **Offer**. Users can open a timeline, read the supporting email, see copied recipients, and review or correct uncertain associations.

## Live demo — synthetic data

**[Open the deployed Streamlit demo](https://interviewtrack-ai-demo.streamlit.app/)**

The public demo uses **synthetic email data only**. No Gmail login, OAuth credentials, API key, or installation is needed. The first screen offers **Demo** and **Gmail Access**. Click **Start Demo** to explore six fictional interview journeys: four Active applications (including two Offers) and two Rejected applications.

Start with **Lumen Robotics — Machine Learning Engineer** for the complete screening-to-offer journey, including multiple technical rounds and staffing specialists in CC. Each visitor gets a separate temporary SQLite database; corrections affect only that browser session. **Reset demo** restores the original examples.

The **Gmail Access** panel explains how to ask the administrator to add your email as a Google OAuth test user. After approval, real Gmail authorization is completed locally using the setup instructions below. The public demo does not connect to real Gmail accounts.

### Synthetic data included in the project

| File | Contents and purpose |
| --- | --- |
| [`demo/fictional-emails.json`](demo/fictional-emails.json) | Readable export of all 31 synthetic input emails, including sender, To, CC, subject, date, body and thread IDs. |
| [`demo/fictional-inbox.db`](demo/fictional-inbox.db) | Generated SQLite snapshot: 29 stored messages after retrieval filtering, plus reconstructed applications and events. No OAuth tokens or real mailbox data. |
| [`demo/SCENARIOS.md`](demo/SCENARIOS.md) | The six scenarios and their expected outcomes, including job alerts and advertisements that stay out of interview cards. |
| [`interviewtrack/fixtures/mailboxes.py`](interviewtrack/fixtures/mailboxes.py) | Source fixture definitions. `PROFILE_REVIEWER` supplies the synthetic messages used by the hosted demo. |

The hosted demo runs the Python reconstruction pipeline from these source fixtures on each new session. The JSON export and bundled SQLite snapshot are included for inspection; neither needs to be uploaded by the reviewer.

### Streamlit deployment configuration

- Repository: `madhumitha-gv/InterviewTrackAI`
- Branch: `main`
- Main file: `cloud/demo_app.py`
- Python: `3.11`
- Dependencies: `cloud/requirements.txt`
- Secrets: none

## How the logic works

### 1. Read and normalize emails

Gmail Mode uses the account the user authorizes with read-only access. The app parses message subjects, sender and recipient addresses, CC recipients, bodies, dates, thread IDs, and reply references. It stores messages and reconstructed applications locally in SQLite.

Demo Mode supplies fictional emails in Gmail-shaped payloads to the same retrieval, parsing, classification, and reconstruction code. It requires no Google account or API key.

### 2. Identify a real interview process

The implementation uses a multi-step parsing and rule-based classification pipeline:

```text
Gmail message IDs
  → metadata triage
  → full-body retrieval for likely recruiting messages and related threads
  → MIME/header parsing and text cleanup
  → event classification and entity extraction
  → application matching and reconstruction
  → interview-only dashboard projection
```

**Two-pass retrieval — `interviewtrack/gmail/client.py`.** `fetch_messages()` first retrieves headers and snippets and computes a recruitment signal score. It then fetches full bodies for messages above the permissive retrieval threshold (0.12) and other messages in qualifying threads. This retains short replies that would otherwise lack recruiting keywords. Repeated synchronization skips known message IDs; failed full-body requests can retain metadata-only records. Retrieval triage narrows the input and is separate from the final interview-card decision.

**Email parsing — `interviewtrack/parsing.py`.** `raw_from_gmail_payload()` converts Gmail JSON into a `RawMessage`: it decodes base64url content, walks MIME body parts, and reads From, To, CC, Date, Message-ID, In-Reply-To, and References. Python's `email.utils` handles address/date parsing. `parse_message()` produces a `ParsedMessage`, preferring plain text, using BeautifulSoup with Python's `html.parser` for HTML, and falling back to snippets when needed. Cleanup removes quoted reply chains, signatures, and footer patterns while retaining the original raw message. Recoverable parsing issues are recorded in `parse_errors`; the implementation can still use partial messages.

**Event classification — `interviewtrack/classify.py`.** `classify_message()` applies weighted regular expressions and subject templates to the cleaned text. It selects the highest-scoring event type and records matched evidence, fired rules, occurrence state, and a confidence score. Candidate-authored messages are handled separately so a copied recruiter subject does not automatically turn a candidate reply into a new recruiter event. Explicit assessment submissions and withdrawals have their own handling.

**Interview-start gate — `interviewtrack/interview_policy.py` and `service.py`.** Advertising patterns are excluded before reconstruction. After reconstruction, `InterviewTrackService.interview_applications()` checks stored messages with `interview_start_evidence()`. This gate looks for incoming screening/interview invitations or explicit advancement language and excludes application receipts, conditional selection wording, and candidate-authored messages as starting evidence. An application is displayed only if it contains a qualifying start event, and its visible timeline begins at that event. Earlier stored context remains available for reconstruction.

### 3. Connect messages to the correct application

**Extract and normalize identity.** `classify.py` extracts employer, job title, role family, requisition ID, recruiter identity, and sender-domain information. Employer extraction prioritizes explicit client/employer language and labeled company fields before weaker affiliation/domain heuristics. `normalize.py` canonicalizes company names, role titles, and requisition IDs so formatting differences can be compared consistently.

**Reconstruct incrementally.** `reconstruct.py` processes messages chronologically. Each reconstructed application has an `ApplicationContext` containing accumulated identity, threads, reply IDs, contacts, and text features. Borderline short messages can be retained through known recruiting threads or senders; explicit advertisement exclusions are not rescued this way. Retention does not itself establish an interview milestone.

**Compute matching features.** `matching.py` compares each message with candidate applications using native Gmail thread membership, RFC reply references, exact requisition matches, employer and role similarity, recruiter/contact overlap, temporal proximity, and cosine text similarity. `contacts.py` maintains learned contact-to-application relationships. Participants include sender, To, and CC addresses, which helps preserve staffing continuity across handoffs.

**Apply constraints before accepting a match.** Conflicting employers, conflicting requisitions, and user-confirmed separation block automatic association. Role conflicts also block it unless an exact requisition match resolves the discrepancy. A native Gmail thread or exact requisition can trigger an automatic match only without conflicting identity evidence. Automated senders require matching employer and role, or an exact requisition match.

**Score the remaining candidates.** Other cases use a weighted feature score with default thresholds of 0.62 for automatic matching and 0.34 for review, subject to identity and blocker checks. These are heuristic scores, not calibrated probabilities. Text similarity contributes to matching but cannot establish application identity alone. If a recruiter handles several roles at the same employer and the message does not identify a role, reconstruction routes the ambiguity to review instead of automatically choosing one.

**Persist evidence and corrections.** Match decisions retain features, scores, reasons, blockers, and rules fired. User merges, splits, assignments, and identity corrections are stored in SQLite and reapplied during reconstruction. Sharing staffing specialists or CC recipients is therefore supporting evidence rather than proof that two roles belong to the same application.

### 4. Build the timeline and resolve the outcome

`reconstruct.py` converts classifications into `InterviewEvent` records linked to their source Gmail messages. The records include event type, occurrence state, sender, timestamp, subject, evidence text, and confidence. Sorting these events produces the application's timeline, including separate interview rounds across unrelated email threads.

`interviewtrack/status.py` uses `resolve()` to compute stage, stage state, latest activity, milestone, and internal status separately:

- **Stage:** a forward-only ranking over non-canceled stage-bearing events. A later screening message cannot move an application backward from a final interview.
- **Stage state:** explicit occurrence evidence determines whether an event is scheduled, completed, canceled, or unknown. An elapsed scheduled date becomes an unknown outcome; it does not imply completion.
- **Latest activity:** the most recent event updates activity independently of the stage. A candidate follow-up does not create another completed interview round.
- **Outcome:** status-changing events provide offer, rejection, or withdrawal evidence. Messages arriving after closure trigger review rather than silently reopening the process.

Finally, `interview_bucket()` projects internal state into the two dashboard views. Rejected processes appear under **Rejected**; offers remain under **Active** with stage **Offer**. Withdrawn processes are omitted. SQLite stores raw messages, derived applications/events, match decisions, corrections, and sync state under the relevant user ID.

### 5. Use optional local embeddings

The default Docker demo uses deterministic hashed n-gram text features with 384 dimensions. This keeps setup small and avoids downloading a neural model. Optional neural mode uses a local embedding model for richer text similarity.

Similarity contributes to the weighted matching score alongside other evidence and can affect automatic matching. It is not sufficient by itself to merge applications, and conflict checks still apply. Optional LLM review is disabled by default.

## Run with Docker

### Prerequisites

Install Docker Desktop and start it before running the commands below. The first build needs internet access to download the base image and Python dependencies.

Extract the project ZIP, open Terminal in the extracted folder, and confirm that it contains `app.py`, `Dockerfile`, and `compose.yaml`.

### Build and start

```sh
docker compose up --build
```

Wait for Streamlit to start, then open **http://localhost:8501**.

1. Select **Fictional inbox — screening through offer**.
2. Click **Start Demo Mode**.
3. Switch between **Active** and **Rejected**.
4. Expand an application to inspect its interview timeline.
5. Open **View email** to read the fictional message and inspect staffing CC recipients.

The demo works without Gmail authorization. The container stores runtime data in a persistent Docker volume. The bundled fictional SQLite database is also available at `demo/fictional-inbox.db` for inspection; Demo Mode reconstructs the fictional source messages into its runtime database.

To run in the background:

```sh
docker compose up --build -d
docker compose ps
```

To stop:

```sh
docker compose down
```

This keeps the runtime data volume for the next run.

### Check the app is running

```sh
docker compose ps
docker compose logs --tail=50 app
curl --fail http://localhost:8501/_stcore/health
```

The health endpoint should return `ok`. It confirms the server is responding; the demo checks below verify application behavior.

If Docker reports that it cannot connect to the daemon, start Docker Desktop and retry. If port 8501 is already in use, choose another port on macOS/Linux:

```sh
APP_PORT=8502 docker compose up --build
```

Then open **http://localhost:8502**.

### Optional neural mode

Stop the default container, then run:

```sh
docker compose down
docker compose -f compose.yaml -f compose.neural.yaml up --build
```

This installs PyTorch and the embedding dependencies. The first use downloads the model; later runs reuse the model cache volume. The default demo remains the quickest way to review the app.

## Demo scenarios and expected results

The fictional input contains **31 emails**. Retrieval filtering leaves **29 stored messages**, and the dashboard shows **six interview applications**: **four Active**, including **two Offers**, and **two Rejected**.

| Company and role | Scenario | Expected result |
| --- | --- | --- |
| Juniper Labs — AI Engineer | Screening invitation and candidate clarification | Active |
| Harbor Energy — Data Engineer | Screening, technical interview, assignment, submission, final interview, follow-up, offer | Active, stage Offer |
| Harbor Energy — Platform Engineer | Screening followed by rejection | Rejected |
| Lumen Robotics — Machine Learning Engineer | Screening, staffing handoffs, two technical rounds, assignment, submission, final interview, follow-up, offer | Active, stage Offer |
| Lumen Robotics — Backend Engineer | Screening, technical interview, rejection, with the same staffing contacts | Rejected; separate from the Machine Learning Engineer application |
| Summit Analytics — Data Engineer | Screening, two technical rounds, candidate follow-up, using the same staffing team | Active; separate from Lumen Robotics |

The Lumen Robotics Machine Learning Engineer application is the complete screening-to-offer example. Elena Ward, Sasha Reed, and Devon Park coordinate across separate threads, with multiple specialists copied on emails. The employer sends the final interview invitation and offer.

Job alerts, interview-coaching advertisements, and application receipts should not appear as interview cards. Clicking **Refresh demo** should preserve the same applications without duplicating messages.

All demo names, email addresses, companies, dates, identifiers, and message text are fictional. The Gmail-formatted candidate address is a display value, not an authorized account. Employer and staffing addresses use `.example` domains. Real mailbox data and credentials are excluded from the project.

See [`demo/SCENARIOS.md`](demo/SCENARIOS.md) for the scenario guide and [`demo/fictional-emails.json`](demo/fictional-emails.json) for every input email, including To and CC fields.

## Run the tests

Build the default image first, then run:

```sh
docker compose build
docker compose run --rm app python -m pytest -q
```

The tests cover parsing, filtering, application matching, stage resolution, corrections, repeated synchronization, and the fictional interview journeys. Neural-specific tests may skip when only core dependencies are installed.

To test the neural image:

```sh
docker compose -f compose.yaml -f compose.neural.yaml build
docker compose -f compose.yaml -f compose.neural.yaml run --rm app python -m pytest -q
```

The latest full local Python run completed with **404 passed and 1 skipped**. Streamlit checks also verified both dashboard views, expanded timelines, and staffing CC display. Both Compose configurations validate. The Docker image build and container runtime have not yet been verified because the local Docker engine was unavailable during development.

## Local Python alternative

Use Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
EMBEDDINGS_ENABLED=false python -m streamlit run app.py
```

Run the tests from the activated environment:

```sh
python -m pytest -q
```

To inspect the synthetic reconstruction without the UI:

```sh
python scripts/demo.py --hashed
```

## Gmail Mode setup

Use the local Python environment for the desktop browser authorization flow.

1. Create a Google Cloud project and enable the Gmail API.
2. Configure the OAuth consent screen and add your account as a test user when the project is in testing mode.
3. Create an OAuth client of type **Desktop app**.
4. Download its JSON file and save it privately as `data/client_secret.json`.
5. Authorize your account:

```sh
python -m interviewtrack.auth --account your.address@gmail.com
python -m interviewtrack.auth --account your.address@gmail.com --status
```

Start Streamlit, select the authorized account in Gmail Mode, and synchronize the mailbox. Authorization is performed through the terminal and browser; there is no frontend Google sign-in button. The reviewer Docker configuration does not mount Gmail credentials.

The app uses read-only Gmail access. Messages, derived timelines, and authorization files stay in the local runtime storage. Credentials and real databases must remain outside version control. The only database intentionally included in this repository is the synthetic demo database.

## Limits

The detection rules primarily cover English recruiting language. Unfamiliar wording, missing role information, and ambiguous staffing relationships may require review. Synthetic test results demonstrate the included cases rather than guaranteeing accuracy for every mailbox. The two-view design shows offers under Active and omits withdrawn applications.

## Repository

[InterviewTrackAI on GitHub](https://github.com/madhumitha-gv/InterviewTrackAI)

