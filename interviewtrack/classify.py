"""Deterministic recruitment classification and structured extraction.

Design rules enforced here:
  * Deterministic first. No model is consulted to decide *what kind* of email this is.
  * Never invent an employer, job title, or outcome. Unknown stays None.
  * Distinguish scheduled / completed / canceled / merely-mentioned.
  * A scheduled time in the past is NOT evidence the interview happened.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .enums import ClassificationSource, EventType, Occurrence
from .models import Classification, Entities, ParsedMessage
from .parsing import unwrap_soft_breaks
from .interview_policy import is_bulk_opportunity
from .normalize import (
    canonical_company,
    canonical_person,
    canonical_role,
    canonical_requisition,
    company_from_domain,
    domain_of,
    extract_requisition_id,
    is_ats_domain,
    is_freemail_domain,
    is_non_employer_domain,
    is_scheduling_domain,
    is_staffing_domain,
    name_from_email,
    role_family,
)

# --- recruitment gate ----------------------------------------------------------

#: Broad recruitment vocabulary. Intentionally generous: the spec warns against an
#: over-restrictive filter that throws away real recruiting conversations. Precision
#: is recovered later by scoring, not by narrowing this gate.
RECRUITMENT_KEYWORDS = (
    "recruit", "recruiter", "recruiting", "recruitment", "talent acquisition",
    "hiring", "hiring manager", "hiring team", "candidate", "candidacy",
    "application", "applied", "apply", "job", "role", "position", "opening",
    "opportunity", "vacancy", "requisition", "req id",
    "interview", "interviewer", "screen", "screening", "phone screen", "tech screen",
    "onsite", "on-site", "panel", "loop", "round",
    "assessment", "take-home", "take home", "takehome", "coding challenge",
    "case study", "technical exercise", "hackerrank", "codesignal", "codility", "karat",
    "offer", "offer letter", "compensation", "salary", "comp package",
    "resume", "cv", "cover letter", "portfolio",
    "we regret", "unfortunately", "not moving forward", "other candidates",
    "move forward", "next steps", "next step", "availability", "schedule",
    "reschedule", "calendar", "time slot", "slots", "background check",
    "reference check", "onboarding", "start date", "team fit", "culture fit",
    "job description", "jd", "career", "careers", "hr", "people team",
    # Short, high-frequency scheduling/confirmation words. These matter most in the
    # metadata-only triage pass, where only a subject line and snippet are available.
    "call", "chat", "conversation", "intro", "introductory", "invitation", "invite",
    "confirm", "confirming", "confirmed", "scheduled", "rescheduled", "meeting",
    "talent acquisition team", "advancing", "congratulations", "submission",
    "take home", "loop", "panel", "debrief", "feedback", "req id", "req-",
)

_KEYWORD_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(RECRUITMENT_KEYWORDS, key=len, reverse=True)),
    re.IGNORECASE,
)

#: Promotional/job-board blasts that mention jobs but are not a candidate's own process.
_NEWSLETTER_MARKERS = (
    "jobs you may like", "recommended jobs", "new jobs for you", "job alert",
    "jobs matching your", "top picks for you", "weekly digest", "people you may know",
    "your job alert", "hiring now near you",
)

#: Gmail labels that indicate bulk mail.
_BULK_LABELS = frozenset({"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"})


def recruitment_signal_score(msg: ParsedMessage) -> tuple[float, list[str]]:
    """Score how likely this message belongs to a hiring process, with reasons.

    Used as the metadata-first triage gate: messages scoring above a low bar get a
    full body fetch.
    """
    if is_bulk_opportunity(msg.raw.subject, msg.clean_text or msg.raw.snippet):
        return 0.0, ["excluded:bulk_opportunity"]
    reasons: list[str] = []
    score = 0.0
    raw = msg.raw
    subject = raw.subject or ""
    text = f"{subject}\n{msg.clean_text or raw.snippet}"
    domain = domain_of(raw.from_addr)

    hits = {m.group(0).lower() for m in _KEYWORD_RE.finditer(text)}
    if hits:
        score += min(0.55, 0.11 * len(hits))
        reasons.append(f"keywords:{len(hits)}")

    if is_ats_domain(domain):
        score += 0.35
        reasons.append("ats_sender")
    if is_staffing_domain(domain):
        score += 0.30
        reasons.append("staffing_sender")
    if is_scheduling_domain(domain):
        score += 0.12
        reasons.append("scheduling_sender")

    local = (raw.from_addr or "").split("@", 1)[0].lower()
    if any(tok in local for tok in ("recruit", "talent", "careers", "jobs", "hiring", "hr",
                                    "people", "interview", "candidate")):
        score += 0.22
        reasons.append("sender_local_part")

    if re.search(r"\b(interview|screen|offer|application|recruiter|assessment)\b", subject, re.I):
        score += 0.20
        reasons.append("subject_signal")

    low = text.lower()
    if any(mark in low for mark in _NEWSLETTER_MARKERS):
        score -= 0.45
        reasons.append("newsletter_marker")
    if _BULK_LABELS & set(raw.labels):
        score -= 0.20
        reasons.append("bulk_label")

    return max(0.0, min(1.0, score)), reasons


def is_probably_recruitment(msg: ParsedMessage, threshold: float = 0.28) -> bool:
    return recruitment_signal_score(msg)[0] >= threshold


# --- event typing --------------------------------------------------------------

# Each rule: (EventType, weight, compiled pattern). Highest total wins.
# Order within a type does not matter; weights encode specificity.
_RULES: list[tuple[EventType, float, re.Pattern[str]]] = [
    # Rejection — checked with high weight because it overrides stage progression.
    (EventType.REJECTION, 1.0, re.compile(
        r"\b(we (?:have )?decided (?:not|to not)|not (?:be )?mov(?:e|ing) forward|"
        r"will not be moving forward|regret to inform|unfortunately,? (?:we|after|your)|"
        r"other candidates whose|moved forward with (?:other|another)|"
        r"not (?:be )?selected|unable to (?:move|proceed|offer)|"
        r"pursuing other candidates|no longer under consideration|"
        r"decided to pursue other|position has been filled|closing your (?:application|candidacy))\b",
        re.IGNORECASE)),
    (EventType.WITHDRAWAL, 1.0, re.compile(
        r"\b(withdraw(?:ing|n|al)? (?:my|your|from)|i(?:'m| am) (?:going to |)withdraw|"
        r"no longer interested in (?:the|this) (?:role|position)|"
        r"decided to accept (?:an|another) offer (?:elsewhere|from another)|"
        r"please remove me from consideration)\b", re.IGNORECASE)),
    (EventType.OFFER, 1.0, re.compile(
        r"\b(offer letter|we(?:'re| are) (?:thrilled|excited|pleased|delighted) to (?:extend|offer)|"
        r"extend(?:ing)? (?:you )?an offer|formal offer|your offer (?:details|package)|"
        r"compensation package|congratulations[,!].{0,60}(?:offer|join)|welcome to the team)\b",
        re.IGNORECASE)),

    # Assessments
    (EventType.ASSESSMENT_ASSIGNED, 0.95, re.compile(
        r"\b(take[- ]?home (?:assignment|exercise|test|challenge|project)|coding (?:challenge|assessment|test)|"
        r"technical (?:assessment|exercise|challenge)|case study (?:assignment|exercise|for you)|"
        r"(?:hackerrank|codesignal|codility|coderpad|karat) (?:assessment|test|link|invitation)|"
        r"complete (?:the|this) assessment|assessment (?:link|invite|invitation|due)|"
        r"you have been (?:invited|assigned).{0,30}assessment)\b", re.IGNORECASE)),
    (EventType.ASSESSMENT_SUBMITTED, 1.35, re.compile(
        r"\b((?:i(?:'ve| have)|we(?:'ve| have)) (?:just )?(?:submitted|completed|finished|attached|sent)"
        r".{0,40}(?:assessment|assignment|take[- ]?home|case study|exercise|challenge|solution)|"
        r"(?:my |the )?(?:completed|finished) (?:assessment|assignment|take[- ]?home|case study)|"
        r"submission (?:received|confirmed)|here(?:'s| is) my (?:solution|submission|write[- ]?up))\b",
        re.IGNORECASE)),

    # Interview rounds
    (EventType.FINAL_INTERVIEW, 0.95, re.compile(
        r"\b(final (?:round|interview|stage|panel)|onsite (?:interview|loop|panel)|on-site (?:interview|loop)|"
        r"full (?:day |)loop|interview loop|last (?:round|stage)|"
        r"(?:meet|meeting) (?:with )?(?:the )?(?:founders?|ceo|vp|leadership|exec))\b", re.IGNORECASE)),
    (EventType.HIRING_MANAGER_INTERVIEW, 0.95, re.compile(
        r"\b(hiring manager (?:interview|conversation|chat|call|round|screen)|"
        r"interview with (?:the )?hiring manager|manager (?:interview|round)|"
        r"conversation with .{0,40}\b(?:director|vp|head of|manager)\b)\b", re.IGNORECASE)),
    (EventType.TECHNICAL_INTERVIEW, 0.9, re.compile(
        r"\b(technical (?:interview|round|screen(?:ing)?|discussion|conversation)|"
        r"tech(?:nical)? screen|coding (?:interview|round|session)|"
        r"system design (?:interview|round)|live coding|pair programming (?:interview|session)|"
        r"ml (?:design|systems) (?:interview|round)|algorithms? (?:interview|round))\b",
        re.IGNORECASE)),
    (EventType.SCREENING_INVITATION, 0.85, re.compile(
        r"\b((?:we(?:'d| would) like to|we want to|would love to) interview you|"
        r"(?:invite|inviting) you to (?:a |an |)(?:first|initial|round ?1|round one)|"
        r"round (?:1|one) (?:will |)(?:consist|interview)|"
        r"first[- ]stage interview|first round of interviews|"
        r"(?:phone|initial|recruiter|intro(?:ductory)?|first) (?:screen(?:ing)?|call|chat|conversation|interview)|"
        r"screening (?:call|interview|conversation)|"
        r"(?:like|love) to (?:set up|schedule|arrange) (?:a |an |)(?:intro|initial|quick|short|brief|30|15)"
        r".{0,20}(?:call|chat|conversation)|invite you to (?:a |an |)(?:screen|phone|initial))\b",
        re.IGNORECASE)),

    # Scheduling / confirmation
    (EventType.SCHEDULING, 0.7, re.compile(
        r"\b(resched(?:ule|uled|uling)|re-?schedul|"
        r"(?:your |the )?interview (?:has been |is |was )?(?:moved|rescheduled|postponed|shifted)|"
        r"pick a (?:time|slot)|choose a (?:time|slot)|book(?:ing)? (?:a |your )?(?:time|slot|interview)|"
        r"here are (?:some |a few |)(?:times|slots|availabilities)|"
        r"(?:what|share your|send your|let me know your) availability|"
        r"calendar invite|invitation: |scheduling link|calendly|"
        r"does .{0,25}(?:work for you|suit you)|are you (?:free|available) on)\b", re.IGNORECASE)),
    (EventType.SCREENING_CONFIRMATION, 1.05, re.compile(
        r"\b(confirm(?:ed|ing|ation)? (?:your |the |our )?(?:\w+ ){0,2}?"
        r"(?:call|interview|screen(?:ing)?|time|slot|meeting)|"
        r"(?:you(?:'re| are)|we(?:'re| are)) (?:all |)(?:set|confirmed|booked)|"
        r"(?:interview|call|screen) confirmed|looking forward to (?:speaking|our (?:call|chat|conversation)) on)\b",
        re.IGNORECASE)),

    # Outreach & follow-ups
    (EventType.RECRUITER_OUTREACH, 0.6, re.compile(
        r"\b(came across your (?:profile|resume|linkedin|background)|"
        r"found your (?:profile|resume)|your (?:profile|background) (?:stood out|caught my eye|"
        r"is a (?:great|strong) (?:fit|match))|"
        r"i(?:'m| am) (?:a |an |)(?:technical |senior |)recruiter (?:at|with|for)|"
        r"reaching out (?:about|regarding|because)|"
        r"(?:exciting |new |)opportunity (?:at|with|for a)|"
        r"would you be (?:open|interested) (?:to|in)|exploring new (?:roles|opportunities))\b",
        re.IGNORECASE)),
    (EventType.RECRUITER_FOLLOWUP, 0.65, re.compile(
        r"\b(just (?:following|checking) (?:up|in)|following up on (?:my|our|your)|"
        r"wanted to (?:follow up|check in|circle back)|circling back|bumping this|"
        r"(?:any|an) update on (?:my|your) (?:side|end)|"
        r"haven(?:'t| not) heard back|gentle (?:nudge|reminder)|"
        r"we(?:'re| are) still (?:reviewing|finalizing|gathering))\b", re.IGNORECASE)),
]

#: Subject-line templates companies actually use at each hiring stage.
#:
#: Applied to the SUBJECT ONLY. Many real recruiting emails carry the whole signal in
#: the subject and almost nothing in the body — an attachment-only "Interview Details"
#: message, or an ATS notification whose body is boilerplate. Without these, such mail
#: falls through to generic correspondence and cannot advance a stage.
_SUBJECT_TEMPLATES: list[tuple[EventType, float, re.Pattern[str]]] = [
    (EventType.OFFER, 1.2, re.compile(
        r"^(?:your\s+)?offer\b|offer\s+of\s+employment|offer\s+letter|"
        r"\boffer\s+from\b", re.I)),
    # NOTE: there is deliberately NO rejection subject template. "Update on your
    # application" is just as often an advance, and inferring an outcome from a
    # subject line would invent one. Rejections are detected from body language.
    (EventType.ASSESSMENT_ASSIGNED, 1.1, re.compile(
        r"case\s+study|take[\s-]?home|coding\s+(?:assessment|challenge|exercise)|"
        r"technical\s+assessment|\bassessment\b", re.I)),
    (EventType.SCREENING_INVITATION, 1.0, re.compile(
        r"interview\s+availability|availability\s+(?:needed|required|request)|"
        r"let'?s\s+connect|intro(?:ductory)?\s+(?:call|chat)|"
        r"schedule\s+(?:a|your)\s+(?:call|chat|interview)|"
        r"quick\s+(?:call|chat)\b", re.I)),
    (EventType.SCREENING_CONFIRMATION, 1.1, re.compile(
        r"interview\s+confirm|^confirmed\s*:|confirmation\s+of\s+interview|"
        r"your\s+interview\s+is\s+confirmed", re.I)),
    # Low weight on purpose: "Invitation: Technical Interview" names a round, and
    # the round must win over the generic scheduling signal.
    # Only patterns the body rules do NOT already cover. "invitation:" and
    # "rescheduled:" are matched there, and adding them here double-counted,
    # letting a generic scheduling signal outrank a named interview round.
    (EventType.SCHEDULING, 0.45, re.compile(
        r"interview\s+details|^updated\s*:|new\s+time\s+for", re.I)),
    (EventType.OTHER_RECRUITMENT, 0.8, re.compile(
        r"thank\s+you\s+for\s+applying|we\s+received\s+your\s+application|"
        r"your\s+application\s+(?:to|for)\b|application\s+received", re.I)),
]


#: Phrases that mean the event already happened. Required for Occurrence.COMPLETED.
_COMPLETED_RE = re.compile(
    r"\b(thank(?:s| you) (?:so much |again |)for (?:your time |the time |)"
    r"(?:(?:yesterday|today|this morning|this afternoon|last week|on \w+day)|"
    r"(?:speaking|chatting|meeting|talking) with)|"
    r"it was (?:great|a pleasure|lovely|nice) (?:speaking|chatting|meeting|talking|connecting) with|"
    r"(?:great|nice|good) (?:speaking|chatting|talking|connecting) with you|"
    r"(?:following|follow) up (?:on|after) (?:our|my|the|yesterday'?s?|today'?s?)"
    r".{0,30}(?:interview|call|conversation|screen|chat|discussion)|"
    r"(?:after|since) (?:our|my|the) (?:interview|call|conversation|screen|chat)|"
    r"(?:you|we) (?:completed|finished|wrapped up|had)(?: the| your| our)? "
    r"(?:interview|screen(?:ing)?|call|assessment|round)|"
    r"(?:interview|screen(?:ing)?|call|assessment|round) (?:is |has been |was )?(?:complete|completed|done)|"
    r"(?:your |the )?(?:feedback|debrief) from (?:your|the|our)|"
    r"we(?:'ve| have) (?:now |)(?:completed|finished|concluded) (?:your|the|our))\b",
    re.IGNORECASE,
)

_CANCELED_RE = re.compile(
    r"\b(cancel(?:ed|led|ling|ing|lation)?\b|"
    r"(?:interview|call|meeting|screen) (?:has been |is |was )?cancel|"
    r"(?:need|have) to cancel|calling off|no longer (?:going|able) to (?:meet|proceed with the (?:call|interview)))\b",
    re.IGNORECASE,
)

_SCHEDULED_RE = re.compile(
    r"\b(scheduled for|is (?:scheduled|booked|set) (?:for|on)|"
    r"(?:will|shall) (?:take place|be held|happen)|"
    r"look(?:ing)? forward to (?:speaking|meeting|our)|"
    r"(?:on|at) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"invitation:|calendar invite|has been (?:confirmed|booked)|"
    r"we(?:'d| would) like to (?:schedule|set up|arrange|invite)|"
    r"(?:please|kindly) (?:confirm|pick|choose|select))\b",
    re.IGNORECASE,
)

#: Unambiguous reschedule/cancel-and-rebook language.
_RESCHEDULE_RE = re.compile(
    r"\b(resched(?:ule|uled|uling)|re-schedul\w*|"
    r"(?:interview|call|meeting|screen) (?:has been |is |was )?(?:moved|postponed|shifted)|"
    r"(?:moved|postponed|shifted) (?:your|the|our) (?:interview|call|meeting|screen)|"
    r"new time for (?:your|the|our))\b",
    re.IGNORECASE,
)

#: Candidate-authored follow-up (distinguished from recruiter follow-up by sender).
_CANDIDATE_FOLLOWUP_RE = re.compile(
    r"\b(thank(?:s| you)(?: so much| again)? for (?:your time|the time|meeting|speaking|chatting)|"
    r"wanted to (?:follow up|check in|reiterate|thank)|"
    r"following up on (?:our|my|the)|"
    r"(?:i|just) wanted to (?:say|express|reiterate|share)|"
    r"(?:any|an) update on (?:the|my) (?:application|candidacy|process|role)|"
    r"excited about the (?:role|opportunity|position))\b",
    re.IGNORECASE,
)


def _occurrence(text: str, event_type: EventType) -> tuple[Occurrence, str]:
    """Decide scheduled/completed/canceled/mentioned from explicit language only."""
    if m := _CANCELED_RE.search(text):
        return Occurrence.CANCELED, m.group(0)
    if m := _COMPLETED_RE.search(text):
        return Occurrence.COMPLETED, m.group(0)
    if m := _SCHEDULED_RE.search(text):
        return Occurrence.SCHEDULED, m.group(0)
    # Submissions and terminal events are inherently completed acts.
    if event_type in {
        EventType.ASSESSMENT_SUBMITTED, EventType.REJECTION,
        EventType.WITHDRAWAL, EventType.OFFER,
    }:
        return Occurrence.COMPLETED, ""
    return Occurrence.MENTIONED, ""


# --- employer / title extraction ----------------------------------------------

#: A company-name span: capitalized words joined by single spaces. The span itself
#: never crosses a newline, which stops "Northwind Analytics\nHi Avery" captures.
#: `(?-i:...)` keeps the capitalization requirement even inside patterns compiled
#: with re.IGNORECASE — otherwise "at Acme Robotics is scheduled" captures "is".
_CO = r"(?-i:[A-Z][\w&\-]*(?:[ ]+(?:[A-Z][\w&\-]*|of|and|&))*)"

#: Separator allowing ONE line wrap between a cue word and the company name, so
#: "...the Data Scientist role at\nGlobex Labs" still resolves.
_S = r"[ \t]*(?:\n[ \t]*)?"

#: Employer cues in descending priority. Tier 1 names the employer of record, which
#: must beat a staffing recruiter's self-identification ("I am a recruiter with X").
_EMPLOYER_TIERS: list[tuple[int, list[re.Pattern[str]]]] = [
    # Tier 1 — explicit employer of record.
    (1, [
        re.compile(r"\b(?:my|our|the)[ ]+client[ ]+(?:is|:)" + _S + r"(" + _CO + r")", re.I),
        re.compile(r"\bon[ ]behalf[ ]of" + _S + r"(" + _CO + r")", re.I),
        re.compile(r"\bclient[ ]*[:\-]" + _S + r"(" + _CO + r")", re.I),
    ]),
    # Tier 2 — labeled company field (scheduling tools, ATS notifications).
    (2, [
        re.compile(r"\b(?:company|employer|organi[sz]ation|hiring[ ]company)[ ]*[:\-]" + _S
                   + r"(" + _CO + r")", re.I),
    ]),
    # Tier 3 — the role/position is explicitly located at an employer.
    (3, [
        re.compile(r"\b(?:position|role|opening|opportunity|job|team|interview|process)"
                   r"[ ]+(?:at|with)" + _S + r"(" + _CO + r")", re.I),
        re.compile(r"\b(?:applying|applied|application)[ ]+(?:to|for|at)" + _S
                   + r"(?:the" + _S + r")?(" + _CO + r")", re.I),
    ]),
    # Tier 4 — generic affiliation cues.
    (4, [
        re.compile(r"\b(?:recruiter|recruiting|talent(?: acquisition)?|hr|people[ ]team)"
                   r"[ ]+(?:at|for|with)" + _S + r"(" + _CO + r")", re.I),
        re.compile(r"\b(?:here[ ])?at" + _S + r"(" + _CO + r")"),
        re.compile(r"\b(" + _CO + r")[ ]+(?:is[ ]hiring|has[ ]an[ ]opening|recruiting[ ]team|"
                   r"talent[ ]team|careers[ ]team)\b"),
    ]),
]

#: Words that look capitalized but are never a company.
_EMPLOYER_STOPWORDS = frozenset(
    {
        "i", "we", "you", "the", "a", "an", "this", "that", "my", "our", "your", "it",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "am", "pm", "est", "pst", "cst",
        "edt", "pdt", "utc", "zoom", "meet", "teams", "calendly", "linkedin",
        "hi", "hello", "hey", "dear", "thanks", "thank", "best", "regards", "sincerely",
        "senior", "staff", "lead", "please", "if", "and", "or", "but",
        "all", "both", "next", "last", "first", "final", "technical", "screening",
        "interview", "assessment", "offer", "role", "position", "team", "hiring",
        "recruiter", "recruiting", "glad", "great", "looking", "forward", "let", "would",
        "could", "scheduled", "confirmed", "unfortunately", "regret", "congratulations",
        "warm", "kind", "cheers", "sent", "from", "subject", "to", "cc", "re",
        "position:", "company:", "when", "where", "who", "what", "candidate",
        "is", "are", "was", "were", "be", "been", "has", "have", "had", "will",
        "on", "in", "with", "for", "at", "by", "about", "regarding",
    }
)


def _trim_company_span(candidate: str) -> str:
    """Trim a captured span down to the actual company name.

    Regex capture is greedy across capitalized words, so "Northwind Analytics. We" or
    "Northwind Analytics Hi Avery" can be captured. Cut at sentence-ending punctuation
    and at the first stopword token.
    """
    candidate = (candidate or "").strip()
    candidate = re.split(r"[.!?,;:](?:\s|$)", candidate)[0]
    tokens = candidate.replace("\n", " ").split()
    kept: list[str] = []
    for tok in tokens:
        bare = re.sub(r"[^A-Za-z0-9&\-]", "", tok).lower()
        if not bare:
            break
        if bare in _EMPLOYER_STOPWORDS:
            if kept:
                break
            return ""
        kept.append(tok.strip(".,;:"))
        if len(kept) >= 4:
            break
    return " ".join(kept).strip(" .,;:-")


def _plausible_company(candidate: str) -> bool:
    canon = canonical_company(candidate)
    if not canon or len(canon) < 2:
        return False
    tokens = canon.split()
    if not tokens or all(t in _EMPLOYER_STOPWORDS for t in tokens):
        return False
    return True


#: A real job title must contain one of these nouns. Without this guard, patterns
#: like "opportunity - great fit for your background" yield nonsense titles.
_JOB_NOUNS = frozenset(
    {
        "engineer", "scientist", "analyst", "developer", "manager", "architect",
        "researcher", "designer", "consultant", "specialist", "administrator",
        "director", "intern", "programmer", "technician", "strategist",
        "mlops", "sre", "devops",
    }
)

#: Vocabulary of words that appear in job titles. A candidate employer whose every
#: token comes from this vocabulary is a role, not a company: "Data Scientist" is
#: rejected while "Northwind Analytics" and "Initech Data" are kept, because
#: "northwind" and "initech" are distinctive non-role tokens.
_ROLE_VOCAB = _JOB_NOUNS | frozenset(
    {
        "data", "machine", "learning", "artificial", "intelligence", "analytics",
        "software", "backend", "frontend", "fullstack", "full", "stack", "platform",
        "business", "product", "research", "applied", "decision", "quantitative",
        "cloud", "security", "site", "reliability", "infrastructure", "etl",
        "generative", "genai", "llm", "large", "language", "model", "natural",
        "processing", "program", "project", "owner", "technical", "ux", "ui",
        "design", "reporting", "insights", "phd", "mlops",
    }
)


def _looks_like_role_only(candidate: str | None) -> bool:
    """True when every token of `candidate` is job-title vocabulary."""
    canon = canonical_role(candidate)
    if not canon:
        return False
    tokens = set(canon.split())
    return bool(tokens) and tokens <= _ROLE_VOCAB


def _is_job_title(title: str | None) -> bool:
    canon = canonical_role(title)
    if not canon:
        return False
    return bool(set(canon.split()) & _JOB_NOUNS)


def _sender_org_names(msg: ParsedMessage) -> set[str]:
    """Canonical names for the SENDER's own organization.

    When the sender is a staffing agency, ATS or scheduling tool, these are excluded
    as employer candidates — the intermediary is not the hiring employer.
    """
    names: set[str] = set()
    domain = domain_of(msg.raw.from_addr)
    if domain:
        for part in domain.split("."):
            cleaned = canonical_company(part.replace("-", " ").replace("mail", " ").strip())
            if cleaned:
                names.add(cleaned)
    # A display name like "Globex Labs via Greenhouse" contributes "greenhouse".
    if msg.raw.from_name and " via " in msg.raw.from_name.lower():
        names.add(canonical_company(msg.raw.from_name.lower().split(" via ", 1)[1]))
    return {n for n in names if n}


#: Language of an agency that has not yet named its client.
_CONFIDENTIAL_CLIENT_RE = re.compile(
    r"\b(confidential\s+(?:search|client|role|opportunity|opening)|"
    r"(?:not\s+able|unable|can'?t|cannot)\s+to?\s+share\s+the\s+(?:company|client|employer)|"
    r"can'?t\s+(?:yet\s+)?(?:name|disclose|reveal)|"
    r"client'?s?\s+(?:name|identity)\s+(?:is\s+)?confidential|"
    r"undisclosed\s+client|a\s+stealth\s+(?:startup|company))\b",
    re.IGNORECASE,
)


def extract_employer(msg: ParsedMessage) -> tuple[str | None, float, str]:
    """Extract the hiring employer. Returns (name, confidence, source).

    Priority: explicit employer-of-record cues, then labeled company fields, then
    role-location cues, then generic affiliation, then the sender's own corporate
    domain. Returns (None, 0.0, "") rather than guessing.

    When the sender is a staffing/ATS/scheduling intermediary, candidates naming the
    sender's OWN organization are rejected, so "I am a recruiter with Harnham and my
    client is Northwind Analytics" yields Northwind, not Harnham.
    """
    raw = msg.raw
    # Unwrap hard-wrapped lines so a company name split across two lines
    # ("at Meridian\nLabs") is captured whole.
    text = unwrap_soft_breaks(f"{raw.subject}\n{msg.clean_text or raw.snippet}")
    domain = domain_of(raw.from_addr)
    intermediary = (
        is_staffing_domain(domain) or is_ats_domain(domain) or is_scheduling_domain(domain)
    )
    own_names = _sender_org_names(msg) if intermediary else set()

    for tier, patterns in _EMPLOYER_TIERS:
        for pattern in patterns:
            for m in pattern.finditer(text):
                candidate = _trim_company_span(m.group(1))
                if not _plausible_company(candidate):
                    continue
                if canonical_company(candidate) in own_names:
                    continue
                if _looks_like_role_only(candidate):
                    continue  # "Data Scientist" is a role, not an employer
                confidence = {1: 0.92, 2: 0.90, 3: 0.85, 4: 0.75}[tier]
                return candidate, confidence, f"text_tier{tier}"

    # An agency withholding its client names no employer. Using the sending domain
    # here would invent the agency as the employer, which rule 5 forbids — such mail
    # goes to review instead.
    if _CONFIDENTIAL_CLIENT_RE.search(text):
        return None, 0.0, ""

    # An ATS display name is only supporting evidence: require the organization
    # to appear independently as a signature line, and never infer an agency client.
    display = re.fullmatch(r"(.+?)\s+(?:Talent Acquisition|Recruiting|Recruitment)(?: Team)?",
                           raw.from_name or "", re.I)
    if display and not is_staffing_domain(domain):
        candidate = _trim_company_span(display.group(1))
        lines = {canonical_company(line.strip()) for line in (msg.clean_text or "").splitlines() if line.strip()}
        if (canonical_company(candidate) in lines and _plausible_company(candidate)
                and not _looks_like_role_only(candidate)
                and canonical_company(candidate) not in own_names):
            return candidate, 0.85, "display_and_signature"

    if domain and not is_non_employer_domain(domain):
        guess = company_from_domain(domain)
        if guess:
            return guess.title(), 0.7, "sender_domain"

    return None, 0.0, ""


#: Job-title patterns, anchored on recruiting phrasing.
_TITLE_PATTERNS = [
    re.compile(r"\b(?:the|our|a|an|this)\s+([A-Z][\w\-/]*(?:\s+[A-Za-z][\w\-/]*){0,4}?)\s+"
               r"(?:role|position|opening|opportunity|req|vacancy|job)\b"),
    re.compile(r"\b(?:role|position|opening|opportunity|title|job)\s*[:\-–]\s*"
               r"([A-Za-z][\w\-/]*(?:\s+[A-Za-z][\w\-/]*){0,4})\b", re.IGNORECASE),
    re.compile(r"\b(?:for|as|regarding|re:|about)\s+(?:the\s+|our\s+|a\s+|an\s+)?"
               r"([A-Z][\w\-/]*(?:\s+[A-Za-z][\w\-/]*){0,4}?)\s+"
               r"(?:role|position|opening|opportunity|req)\b"),
    re.compile(r"\b(?:hiring|recruiting|interviewing)\s+(?:for\s+)?(?:a|an|the)?\s*"
               r"([A-Z][\w\-/]*(?:\s+[A-Za-z][\w\-/]*){0,4}?)\b"),
]

#: Known title keywords used as a last-resort scan.
_TITLE_KEYWORD_RE = re.compile(
    r"\b("
    # Known domain modifiers before a core job noun.
    r"(?:(?:senior|sr\.?|staff|principal|lead|junior|jr\.?|associate)\s+)?"
    r"(?:data|machine learning|ml|ai|artificial intelligence|software|backend|frontend|"
    r"full[- ]stack|research|applied|analytics|business|product|platform|devops|"
    r"site reliability|cloud|security|genai|generative ai|llm|quantitative|clinical|"
    r"perception|mlops|nlp)\s+"
    r"(?:scientist|engineer(?:ing)?|analyst|analytics|developer|manager|architect|"
    r"researcher|intelligence\s+\w+)"
    r"(?:\s+(?:i{1,3}|iv|\d))?"
    r")\b",
    re.IGNORECASE,
)


#: Language that redirects the message to a DIFFERENT role than the one named
#: earlier. A reply subject carries the old topic, and the body often names the old
#: role before the new one, so without this the stale role wins.
_ROLE_SWITCH_RE = re.compile(
    r"\b(instead|rather than|different (?:role|position|team|opening)|"
    r"another (?:role|position|opening)|changing topic|in place of|"
    r"headcount (?:was |is )?frozen|no longer (?:open|available)|"
    r"put you forward for)\b",
    re.IGNORECASE,
)


def _titles_in(text: str) -> list[tuple[int, str]]:
    """Every job title in `text` with its character offset."""
    out: list[tuple[int, str]] = []
    for m in _TITLE_KEYWORD_RE.finditer(text or ""):
        title = re.sub(r"\s+", " ", m.group(1)).strip(" .,;:-")
        if _is_job_title(title):
            out.append((m.start(), title))
    return out


def extract_job_title(msg: ParsedMessage) -> str | None:
    """Extract the job title, or None. Never fabricates.

    When the message explicitly switches roles ("the X headcount was frozen —
    instead, our Y position"), the role named AFTER the switch wins. Otherwise the
    first stated title wins.
    """
    raw = msg.raw
    body = unwrap_soft_breaks(msg.clean_text or raw.snippet)

    titles = _titles_in(body)
    distinct = {canonical_role(t) for _, t in titles}
    if len(distinct) > 1 and _ROLE_SWITCH_RE.search(body):
        # The message names more than one role AND explicitly redirects between
        # them. The replacement role is stated last ("the X headcount was frozen —
        # instead ... our Y position"), so the last distinct title wins. Cue position
        # is deliberately NOT used: cues like "changing topic" open the sentence that
        # names the OLD role.
        return titles[-1][1]

    # Subject lines carry the title most reliably for a non-reply.
    for source in (raw.subject, msg.clean_text or raw.snippet):
        if not source:
            continue
        if m := _TITLE_KEYWORD_RE.search(source):
            title = re.sub(r"\s+", " ", m.group(1)).strip(" .,;:-")
            if _is_job_title(title):
                return title
    for source in (raw.subject, msg.clean_text or raw.snippet):
        if not source:
            continue
        for pattern in _TITLE_PATTERNS:
            if m := pattern.search(source):
                title = re.sub(r"\s+", " ", m.group(1)).strip(" .,;:-")
                if _is_job_title(title):
                    return title
    return None


# --- event datetime ------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_RE = re.compile(
    r"\b(?:(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?,?\s+)?"
    r"(?:(" + "|".join(_MONTHS) + r")\w*\s+(\d{1,2})|(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\w*)"
    r"(?:,?\s+(\d{4}))?"
    r"(?:\s*(?:at|@|,)?\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?))?",
    re.IGNORECASE,
)


def extract_event_datetime(text: str, reference: datetime | None) -> datetime | None:
    """Parse an explicit interview date/time from text.

    `reference` supplies the year when the text omits it. Returns None when nothing
    explicit is present — the message date is never used as a substitute.
    """
    if not text:
        return None
    m = _DATE_RE.search(text)
    if not m:
        return None
    mon_name_a, day_a, day_b, mon_name_b, year_s, hour_s, minute_s, ampm = m.groups()
    mon_name = (mon_name_a or mon_name_b or "").lower()
    day_s = day_a or day_b
    if not mon_name or not day_s:
        return None
    month = _MONTHS.get(mon_name)
    if not month:
        return None
    try:
        day = int(day_s)
        year = int(year_s) if year_s else (reference.year if reference else datetime.now().year)
        hour, minute = 0, 0
        if hour_s:
            hour = int(hour_s) % 12
            minute = int(minute_s) if minute_s else 0
            if ampm and ampm.lower().startswith("p"):
                hour += 12
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return None


# --- top-level classification --------------------------------------------------


def _is_from_candidate(msg: ParsedMessage, user_email: str | None) -> bool:
    """Did the user send this message? Falls back to the SENT label."""
    if user_email and msg.raw.from_addr:
        if msg.raw.from_addr.lower() == user_email.lower():
            return True
    return "SENT" in msg.raw.labels


def classify_message(msg: ParsedMessage, user_email: str | None = None) -> Classification:
    """Classify one parsed message into a typed, evidence-backed event."""
    raw = msg.raw
    subject = raw.subject or ""
    body = msg.clean_text or raw.snippet or ""
    text = f"{subject}\n{body}"
    if is_bulk_opportunity(subject, body):
        return Classification(gmail_id=raw.gmail_id, thread_id=raw.thread_id,
                              event_type=EventType.NOT_RECRUITMENT, is_recruitment=False,
                              confidence=1.0, rules_fired=["excluded:bulk_opportunity"])
    domain = domain_of(raw.from_addr)
    from_candidate = _is_from_candidate(msg, user_email)

    signal, signal_reasons = recruitment_signal_score(msg)
    rules_fired = list(signal_reasons)

    # Accumulate rule weights per event type.
    scores: dict[EventType, float] = {}
    evidence_by_type: dict[EventType, str] = {}
    for event_type, weight, pattern in _RULES:
        if m := pattern.search(text):
            # A subject-line hit is more decisive than a body hit.
            bonus = 0.15 if pattern.search(subject) else 0.0
            scores[event_type] = scores.get(event_type, 0.0) + weight + bonus
            evidence_by_type.setdefault(event_type, m.group(0))
            rules_fired.append(f"rule:{event_type.value}")

    # --- subject-line templates ---
    # Skipped for mail the candidate SENT: a reply keeps the recruiter's subject,
    # so "Re: Your take-home assignment" describes what was *received*, not that the
    # candidate is assigning anything. Their own body says what they did.
    for event_type, weight, pattern in ([] if from_candidate else _SUBJECT_TEMPLATES):
        if m := pattern.search(subject):
            scores[event_type] = scores.get(event_type, 0.0) + weight
            evidence_by_type.setdefault(event_type, m.group(0))
            rules_fired.append(f"subject:{event_type.value}")

    # An explicit reschedule is about *scheduling*, whatever round it concerns.
    if _RESCHEDULE_RE.search(text) and scores:
        scores[EventType.SCHEDULING] = scores.get(EventType.SCHEDULING, 0.0) + 0.6
        rules_fired.append("rule:explicit_reschedule")

    if scores:
        event_type = max(scores.items(), key=lambda kv: kv[1])[0]
        rule_confidence = min(0.95, 0.5 + 0.22 * scores[event_type])
        evidence = evidence_by_type.get(event_type, "")
    else:
        event_type = EventType.OTHER_RECRUITMENT if signal >= 0.28 else EventType.NOT_RECRUITMENT
        rule_confidence = 0.4 if signal >= 0.28 else 0.15
        evidence = ""

    # Candidate-authored mail is a candidate follow-up, not recruiter activity —
    # unless it is an explicit submission or withdrawal.
    if from_candidate and event_type not in {
        EventType.ASSESSMENT_SUBMITTED, EventType.WITHDRAWAL,
    }:
        if _CANDIDATE_FOLLOWUP_RE.search(text) or event_type in {
            EventType.RECRUITER_FOLLOWUP, EventType.OTHER_RECRUITMENT,
            EventType.RECRUITER_OUTREACH, EventType.SCHEDULING,
        }:
            if m := _CANDIDATE_FOLLOWUP_RE.search(text):
                evidence = m.group(0)
            event_type = EventType.CANDIDATE_FOLLOWUP
            rules_fired.append("rule:candidate_authored")

    occurrence, occ_evidence = _occurrence(text, event_type)

    # Guard rail: a *mention* of a past interview in a follow-up must not create a
    # new completed interview round.
    if event_type == EventType.CANDIDATE_FOLLOWUP:
        occurrence = Occurrence.COMPLETED if occ_evidence else Occurrence.MENTIONED

    # --- structured extraction ---
    employer, emp_conf, emp_source = extract_employer(msg)
    title = extract_job_title(msg)
    req = extract_requisition_id(subject, body)

    recruiter_name = canonical_person(raw.from_name) or name_from_email(raw.from_addr)
    if from_candidate:
        recruiter_name, recruiter_email = "", None
    else:
        recruiter_email = raw.from_addr or None

    entities = Entities(
        employer=employer,
        employer_canonical=canonical_company(employer) or None,
        employer_confidence=emp_conf,
        employer_source=emp_source,
        job_title=title,
        job_title_canonical=canonical_role(title) or None,
        role_family=role_family(title),
        requisition_id=canonical_requisition(req) or None,
        recruiter_name=recruiter_name or None,
        recruiter_email=recruiter_email,
        sender_domain=domain,
        is_staffing_sender=is_staffing_domain(domain),
        is_ats_sender=is_ats_domain(domain),
    )

    event_dt = extract_event_datetime(text, raw.date)

    # Confidence blends rule strength with recruitment signal and parse health.
    confidence = round(min(0.97, 0.6 * rule_confidence + 0.4 * max(signal, rule_confidence)), 4)
    if msg.parse_errors:
        confidence = round(confidence * 0.8, 4)

    return Classification(
        gmail_id=raw.gmail_id,
        thread_id=raw.thread_id,
        event_type=event_type,
        occurrence=occurrence,
        is_recruitment=event_type != EventType.NOT_RECRUITMENT,
        entities=entities,
        event_datetime=event_dt,
        evidence=(evidence or occ_evidence or (body[:200] if body else subject))[:300],
        confidence=confidence,
        source=ClassificationSource.DETERMINISTIC,
        rules_fired=sorted(set(rules_fired)),
    )
