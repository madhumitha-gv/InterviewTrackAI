"""InterviewTrack AI — Streamlit application.

Deliberately minimal per the spec: company/application cards and reconstructed
interview timelines. No dashboard metrics, analytics charts or KPI summaries, and no
decorative controls — every button does something.

Run:  streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import streamlit as st

from interviewtrack.config import get_settings
from interviewtrack.db import get_db
from interviewtrack.embeddings import get_backend
from interviewtrack.enums import ApplicationStatus, Occurrence
from interviewtrack.gmail import desktop_oauth
from interviewtrack.sources import (
    GmailEmailSource,
    SyntheticEmailSource,
    gmail_available,
)
from interviewtrack.gmail.links import best_message_link, thread_link
from interviewtrack.models import Application
from interviewtrack.service import (
    DEMO_USER_ID,
    FILTERS,
    InterviewTrackService,
    search_applications,
    sort_applications,
)
from interviewtrack.status import event_label
from interviewtrack.interview_policy import interview_bucket

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="InterviewTrack AI", page_icon="📬", layout="wide")

SETTINGS = get_settings()

STATUS_BADGE = {
    ApplicationStatus.ACTION_REQUIRED: ("🟠", "Action required"),
    ApplicationStatus.SCHEDULED: ("🔵", "Scheduled"),
    ApplicationStatus.AWAITING_RESPONSE: ("⏳", "Awaiting response"),
    ApplicationStatus.OFFER: ("🎉", "Offer"),
    ApplicationStatus.REJECTED: ("⚪", "Rejected"),
    ApplicationStatus.WITHDRAWN: ("⚪", "Withdrawn"),
    ApplicationStatus.NEEDS_REVIEW: ("🟡", "Needs review"),
}

OCCURRENCE_MARK = {
    Occurrence.COMPLETED: "✅ completed",
    Occurrence.SCHEDULED: "🗓 scheduled",
    Occurrence.CANCELED: "❌ canceled",
    Occurrence.MENTIONED: "• mentioned",
}


# --- session helpers -----------------------------------------------------------


@st.cache_resource
def _service() -> InterviewTrackService:
    """One service (and one embedding model) per Streamlit process."""
    return InterviewTrackService(db=get_db(), settings=SETTINGS, backend=get_backend())


def _ss(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _signed_in() -> bool:
    return bool(st.session_state.get("user_id"))


def _demo_mode() -> bool:
    uid = st.session_state.get("user_id", "")
    return uid == DEMO_USER_ID or uid.startswith("demo:")


# --- authentication ------------------------------------------------------------


def render_login(svc: InterviewTrackService) -> None:
    st.title("📬 InterviewTrack AI")
    st.caption("Your interviews, follow-ups and offers in one place.")
    st.divider()

    demo_col, gmail_col = st.columns(2, gap="large")

    with demo_col:
        st.subheader("Demo Mode")
        st.caption("Explore fictional emails. No Gmail connection needed.")
        demo_choices = {
            "Fictional inbox — screening through offer": "reviewer",
            "Full scenario set (32 emails, 23 threads)": None,
            "Profile 1 — Conventional inbox": "conventional",
            "Profile 2 — Unfamiliar domains & custom labels": "unfamiliar",
            "Profile 3 — High-volume inbox, sparse recruitment": "sparse",
        }
        choice = st.selectbox(
            "Dataset", list(demo_choices), key="demo_profile",
            help="The three profiles show the same code handling different inbox "
                 "structures (unknown agencies, odd domains, heavy noise).",
        )
        if st.button("Start Demo Mode", type="primary", use_container_width=True):
            profile_key = demo_choices[choice]
            with st.spinner("Running the pipeline…"):
                outcome = svc.load_demo_data(profile_key=profile_key)
            if profile_key:
                from interviewtrack.fixtures.mailboxes import ALL_PROFILES
                prof = next(p for p in ALL_PROFILES if p.key == profile_key)
                uid, email, name = f"demo:{profile_key}", prof.user_email, prof.label
            else:
                uid, email, name = DEMO_USER_ID, "avery.candidate@gmail.com", "Avery (Demo)"
            st.session_state.update(
                user_id=uid, user_email=email, user_name=name,
                mode="demo", demo_profile_key=profile_key, last_sync=outcome,
            )
            st.rerun()

    with gmail_col:
        st.subheader("Gmail Mode")
        accounts = desktop_oauth.stored_accounts(SETTINGS)
        choices = [a for a in accounts if not a.startswith("(legacy")]
        if any(a.startswith("(legacy") for a in accounts):
            choices.insert(0, "Legacy account")
        selected = st.selectbox("Google account", choices) if choices else None
        account_key = None if selected == "Legacy account" else selected
        authorized, reason = gmail_available(account_key, SETTINGS)
        if authorized:
            account = desktop_oauth.account_email(account_key, SETTINGS)
            if not account:
                st.error("Unable to verify the connected account. Reauthorize before syncing.")
                return
            st.success(f"Authorized as **{account}** (read-only)")
            st.caption(
                f"Reads the last {SETTINGS.lookback_days} days of mail — inbox, sent "
                "and archived, excluding Spam and Trash. Nothing is ever modified."
            )
            if st.button("Connect to Gmail", type="primary", use_container_width=True):
                svc.register_user(f"gmail:{account}", account, account)
                st.session_state.update(
                    gmail_account=account_key,
                    user_id=f"gmail:{account}", user_email=account,
                    user_name=account, mode="gmail",
                )
                st.rerun()
        else:
            st.warning("Not authorized yet.")
            st.caption(
                "Gmail Mode needs a one-time authorization from your terminal. It is "
                "deliberately not run from this page: Streamlit re-runs its whole "
                "script on every interaction, which would re-trigger OAuth repeatedly."
            )
            st.code("python -m interviewtrack.auth", language="bash")
            with st.expander("What the error says"):
                st.text(reason)
            st.caption("See README → Gmail Mode setup for the Google Cloud steps.")



# --- sidebar -------------------------------------------------------------------


def render_sidebar(svc: InterviewTrackService) -> None:
    user_id = st.session_state["user_id"]
    with st.sidebar:
        st.markdown(f"**{st.session_state.get('user_name', '')}**")
        st.caption(st.session_state.get("user_email", ""))
        if st.session_state.get("mode") == "gmail":
            st.success("Gmail Mode — read-only")
        else:
            st.info("Demo Mode — synthetic data, no Gmail account connected.")

        st.divider()
        if st.button("🔄 Refresh demo" if _demo_mode() else "🔄 Refresh Gmail", use_container_width=True):
            refresh(svc)

        state = svc.sync_state(user_id)
        if state and state.get("last_sync_at"):
            st.caption(f"Last sync: {state['last_sync_at'][:19].replace('T', ' ')} UTC")
            st.caption(f"Messages stored: {state.get('message_count', 0)}")

        st.divider()
        if st.button("🚪 Log out", use_container_width=True):
            logout(svc)

        with st.expander("Delete my local data"):
            st.caption(
                "Removes every stored message, application, decision and correction "
                "for this account from the local SQLite database. The locally stored "
                "Google authorization is retained; use the auth --revoke command to remove it."
            )
            if st.button("Delete everything", type="primary", use_container_width=True):
                counts = svc.delete_everything(user_id)
                st.session_state.clear()
                st.success(f"Deleted: {counts}")
                st.rerun()


def refresh(svc: InterviewTrackService) -> None:
    """Re-sync through whichever source this session is using."""
    user_id = st.session_state["user_id"]
    email = st.session_state.get("user_email", "")
    mode = st.session_state.get("mode", "demo")
    now = datetime(2025, 10, 28, tzinfo=timezone.utc) if mode == "demo" else None

    if mode == "demo":
        st.session_state["last_sync"] = svc.load_demo_data(
            profile_key=st.session_state.get("demo_profile_key"))
        st.rerun()
        return
    source = GmailEmailSource(account=st.session_state.get("gmail_account"))
    bar = st.progress(0.0, text=f"Retrieving from {source.name}…")

    def progress(done: int, total: int) -> None:
        bar.progress(min(1.0, done / max(1, total)),
                     text=f"Fetched {done}/{total} messages…")

    outcome = svc.sync_from_source(user_id, email, source, progress=progress, now=now)
    bar.empty()
    st.session_state["last_sync"] = outcome
    if outcome.errors and mode == "gmail":
        # Gmail problems are surfaced, never papered over with synthetic data.
        st.error(outcome.errors[0])
        return
    st.rerun()


def logout(svc: InterviewTrackService) -> None:
    # Logging out clears the session only. The Gmail token stays on disk so the
    # user is not forced through authorization again; `python -m interviewtrack.auth
    # --revoke` removes it.
    st.session_state.clear()
    st.rerun()


# --- application cards ---------------------------------------------------------


def render_timeline(svc: InterviewTrackService, app: Application) -> None:
    st.markdown("**Reconstructed interview timeline**")
    messages = {m.gmail_id: m for m in svc.db.load_messages(st.session_state["user_id"])}
    for ev in app.timeline:
        when = ev.occurred_at.strftime("%b %d, %Y") if ev.occurred_at else "Unknown date"
        mark = OCCURRENCE_MARK.get(ev.occurrence, "")
        sender = ev.sender_name or ev.sender or "unknown sender"

        with st.container(border=True):
            top, link = st.columns([5, 1])
            with top:
                st.markdown(f"**{when} — {event_label(ev.event_type)}**  ·  {mark}")
                st.markdown(f"*{ev.subject or '(no subject)'}*")
                st.caption(f"From {sender} <{ev.sender}>")
                message = messages.get(ev.gmail_id)
                if message:
                    if message.cc_addrs:
                        st.caption("CC: " + ", ".join(message.cc_addrs))
                    with st.expander("View email"):
                        st.text(message.body_text or message.snippet)
                if ev.preview:
                    st.text(ev.preview[:240] + ("…" if len(ev.preview) > 240 else ""))
                if ev.event_datetime:
                    st.caption(
                        f"Event date stated in the email: "
                        f"{ev.event_datetime.strftime('%b %d, %Y %H:%M UTC')}"
                    )
                if ev.evidence:
                    st.caption(f"Evidence: “{ev.evidence[:160]}”")
            with link:
                if not _demo_mode():
                    st.link_button(
                        "Open in Gmail",
                        best_message_link(ev.rfc_message_id, ev.thread_id, st.session_state.get("user_email")),
                        use_container_width=True,
                    )



def render_evidence(svc: InterviewTrackService, app: Application) -> None:
    """Why these threads were associated — the audit trail for every merge.

    Decisions live in their own SQLite table, so they are read back from there
    rather than from the in-memory Application object.
    """
    import json as _json

    user_id = st.session_state["user_id"]
    rows = [
        r for r in svc.db.load_decisions(user_id)
        if r.get("app_key") == app.key and r.get("kind") != "NO_MATCH"
    ]
    if not rows:
        st.caption(
            "No cross-thread association was needed for this application — every "
            "message arrived in a single Gmail thread."
        )
        return

    st.caption(
        f"{len(rows)} association decision(s). Each merge below is backed by explicit "
        "evidence; semantic similarity alone can never merge two applications."
    )
    for r in rows:
        st.markdown(
            f"`{r['gmail_id']}` → **{r['kind']}** "
            f"(score {r['score']:.2f}, source: {r['source']})"
        )
        for reason in _json.loads(r.get("reasons_json") or "[]")[:6]:
            st.caption(f"• {reason}")
        blockers = _json.loads(r.get("blockers_json") or "[]")
        if blockers:
            st.caption("⛔ blockers: " + "; ".join(blockers))
        try:
            f = _json.loads(r.get("features_json") or "{}")
            st.caption(
                f"features — requisition_match={f.get('requisition_match')} "
                f"employer_match={f.get('employer_match')} "
                f"role_similarity={float(f.get('role_similarity', 0)):.2f} "
                f"semantic_similarity={float(f.get('semantic_similarity', 0)):.2f} "
                f"same_gmail_thread={f.get('same_gmail_thread')} "
                f"shared_recruiter={f.get('shared_recruiter')} "
                f"days_apart={float(f.get('days_apart', 0)):.0f}"
            )
        except (ValueError, TypeError):
            pass
        st.divider()


def render_corrections(svc: InterviewTrackService, app: Application,
                       all_apps: list[Application]) -> None:
    """Merge / split / field corrections. Persisted and re-applied on every sync."""
    user_id = st.session_state["user_id"]
    email = st.session_state.get("user_email", "")
    now = datetime(2025, 10, 28, tzinfo=timezone.utc) if _demo_mode() else None

    st.markdown("**Merge into another application**")
    others = [a for a in all_apps if a.key != app.key]
    if others:
        labels = {f"{a.display_title}  ({a.thread_count} threads)": a.key for a in others}
        choice = st.selectbox("Target application", list(labels),
                              key=f"merge_target_{app.key}")
        if st.button("Merge this application into the target", key=f"merge_btn_{app.key}"):
            svc.merge_applications(user_id, email, app.key, labels[choice], now=now)
            st.success("Merged. The correction is stored and will survive future syncs.")
            st.rerun()
    else:
        st.caption("No other application to merge into.")

    st.divider()
    st.markdown("**Split a message out of this application**")
    if len(app.events) > 1:
        ev_labels = {
            f"{e.gmail_id} — {e.subject[:52] or '(no subject)'}": e.gmail_id
            for e in app.timeline
        }
        pick = st.selectbox("Message to detach", list(ev_labels), key=f"split_pick_{app.key}")
        if st.button("Detach into its own application", key=f"split_btn_{app.key}"):
            svc.split_message(user_id, email, app.key, ev_labels[pick], now=now)
            st.success("Detached into a separate application.")
            st.rerun()
    else:
        st.caption("Needs at least two messages to split.")

    st.divider()
    st.markdown("**Correct employer / role / requisition**")
    c1, c2, c3 = st.columns(3)
    emp = c1.text_input("Employer", value=app.employer, key=f"emp_{app.key}")
    title = c2.text_input("Job title", value=app.job_title or "", key=f"title_{app.key}")
    req = c3.text_input("Requisition ID", value=app.requisition_id or "", key=f"req_{app.key}")
    if st.button("Save corrections", key=f"fields_btn_{app.key}"):
        svc.correct_fields(user_id, email, app.key, employer=emp or None,
                           job_title=title or None, requisition_id=req or None, now=now)
        st.success("Saved. Corrections persist across synchronizations.")
        st.rerun()


def render_card(svc: InterviewTrackService, app: Application,
                all_apps: list[Application], expanded: bool) -> None:
    status_text = interview_bucket(app) or "Active"
    icon = "⚪" if status_text == "Rejected" else "🟢"

    with st.container(border=True):
        head, meta = st.columns([3, 2])
        with head:
            st.markdown(f"### {app.employer}")
            st.markdown(f"**{app.job_title or 'Role not stated in available emails'}**")
            if app.requisition_id:
                st.caption(f"Requisition `{app.requisition_id}`")
        with meta:
            st.markdown(f"{icon} **{status_text}**")
            stage_bits = [f"Current stage: **{app.current_stage.value.replace('_', ' ')}**"]
            if app.stage_state.value != "unknown":
                stage_bits.append(f"({app.stage_state.value})")
            st.markdown(" ".join(stage_bits))
            if app.stage_note:
                st.caption(f"⚠️ {app.stage_note}")
            st.caption(
                f"Latest activity: {app.latest_activity_label or '—'}"
                + (f" · {app.latest_activity_at.strftime('%b %d, %Y')}"
                   if app.latest_activity_at else "")
            )
            st.caption(f"Last supported milestone: {app.last_supported_milestone or '—'}")

        c1, c2, c3 = st.columns(3)
        c1.caption(f"**Gmail threads:** {app.thread_count}")
        c2.caption(f"**Timeline events:** {len(app.events)}")
        c3.caption(f"**Match confidence:** {app.match_confidence:.2f}")

        if app.contacts:
            st.caption(
                "**Contacts:** "
                + ", ".join(
                    f"{c.label} — {c.contact_type.value.replace('_', ' ')}"
                    for c in app.contacts[:6]
                )
            )

        if app.needs_review:
            st.warning("⚠️ Ambiguous association — human review requested.")
        if app.history_possibly_incomplete:
            st.info(
                "ℹ️ History may be incomplete: the three-month retrieval window may "
                "not include earlier events."
            )
        for note in app.notes:
            st.caption(f"📝 {note}")

        with st.expander("Interview timeline, evidence and corrections", expanded=expanded):
            tabs = st.tabs(["Timeline", "Why these threads matched", "Threads", "Correct"])
            with tabs[0]:
                render_timeline(svc, app)
            with tabs[1]:
                render_evidence(svc, app)
            with tabs[2]:
                for tid in sorted(set(app.thread_ids)):
                    n = sum(1 for e in app.events if e.thread_id == tid)
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"Thread `{tid}` — {n} message(s)")
                    cols[1].link_button("Open thread", thread_link(tid, st.session_state.get("user_email")),
                                        use_container_width=True)
            with tabs[3]:
                render_corrections(svc, app, all_apps)


def render_review_queue(svc: InterviewTrackService, apps: list[Application]) -> None:
    queue = [a for a in apps if a.needs_review]
    if not queue:
        return
    with st.expander(f"⚠️ Ambiguous matches needing review ({len(queue)})", expanded=True):
        st.caption(
            "These were NOT merged automatically. Nothing was invented: either the "
            "employer could not be identified, or the evidence was too weak to "
            "associate the messages with an existing application."
        )
        user_id = st.session_state["user_id"]
        email = st.session_state.get("user_email", "")
        now = datetime(2025, 10, 28, tzinfo=timezone.utc) if _demo_mode() else None

        for app in queue:
            st.divider()
            st.markdown(f"**{app.display_title}** — {len(app.events)} message(s)")
            for note in app.notes:
                st.caption(f"📝 {note}")
            for ev in app.timeline[:3]:
                st.caption(
                    f"• {ev.occurred_at.strftime('%b %d') if ev.occurred_at else '?'} "
                    f"— {ev.subject[:70]}"
                )

            cols = st.columns([3, 1])
            others = [a for a in apps if a.key != app.key]
            if others:
                labels = {a.display_title: a.key for a in others}
                target = cols[0].selectbox(
                    "Merge into", list(labels), key=f"rq_target_{app.key}"
                )
                if cols[0].button("Confirm merge", key=f"rq_merge_{app.key}"):
                    svc.merge_applications(user_id, email, app.key, labels[target], now=now)
                    st.rerun()
            if cols[1].button("Keep separate", key=f"rq_keep_{app.key}"):
                svc.confirm_review(user_id, email, app.key, now=now)
                st.rerun()


# --- main ----------------------------------------------------------------------


def main() -> None:
    svc = _service()

    if not _signed_in():
        render_login(svc)
        return

    user_id = st.session_state["user_id"]
    email = st.session_state.get("user_email", "")

    # Gmail Mode pulls on first entry; Demo Mode is already seeded.
    if (st.session_state.get("mode") == "gmail"
            and not st.session_state.get("initial_sync_attempted")
            and not svc.sync_state(user_id)):
        st.session_state["initial_sync_attempted"] = True
        refresh(svc)

    render_sidebar(svc)

    st.title("📬 InterviewTrack AI")
    st.caption("Track each role from the first interview to the final outcome.")
    apps = svc.interview_applications(user_id, email)

    outcome = st.session_state.get("last_sync")
    if outcome is not None and outcome.errors:
        with st.expander(f"⚠️ {len(outcome.errors)} issue(s) during the last sync"):
            for err in outcome.errors[:20]:
                st.caption(err)

    if not apps:
        st.info(
            "No interview invitations or explicit next-step invitations found yet. Job alerts and application receipts are hidden. Use **Refresh Gmail** in "
            "the sidebar to synchronize the last three months of mail."
        )
        return

    top = st.columns([3, 2, 1])
    query = top[0].text_input("Search by company, position or recruiter",
                              placeholder="e.g. Northwind, Data Scientist, Priya")
    filter_names = ["Active", "Rejected"]
    chosen = top[1].radio(
        "Filter", filter_names, horizontal=True,
        index=filter_names.index("Active"),   # name, not a positional index
        label_visibility="visible",
    )
    expand_all = top[2].toggle("Expand all", value=False)

    render_review_queue(svc, apps)

    visible = sort_applications(
        [a for a in search_applications(apps, query) if interview_bucket(a) == chosen]
    )
    st.caption(
        f"Showing **{len(visible)}** of {len(apps)} reconstructed application(s) "
        f"across {sum(a.thread_count for a in apps)} Gmail thread(s)."
    )
    if not visible:
        st.info("No applications match the current search and filter.")

    for app in visible:
        render_card(svc, app, apps, expanded=expand_all)


if __name__ == "__main__":
    main()
