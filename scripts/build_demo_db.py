"""Create a fresh fictional SQLite database; never reads a real mailbox."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interviewtrack.db import Database
from interviewtrack.service import InterviewTrackService
from interviewtrack.embeddings import get_backend
root = Path(__file__).resolve().parent.parent
path = root / "demo" / "fictional-inbox.db"
if path.exists():
    raise SystemExit("Demo database already exists; choose a fresh destination before rebuilding.")
db = Database(path)
svc = InterviewTrackService(db=db, backend=get_backend(force_hashed=True))
result = svc.load_demo_data(profile_key="reviewer")
apps = svc.interview_applications("demo:reviewer", "avery.quinn.interviewtrack.demo@gmail.com")
from interviewtrack.interview_policy import interview_bucket
for app in apps:
    print(app.display_title, interview_bucket(app), app.current_stage.value, len(app.events))
assert len(apps) == 6
assert sum(interview_bucket(a) == "Rejected" for a in apps) == 2
assert any(a.current_stage.value == "offer" for a in apps)
assert not result.errors, result.errors
db.close()
print(path)
