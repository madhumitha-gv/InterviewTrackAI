"""Stage B — entity normalization.

Canonicalizes employers, role titles, requisition IDs and recruiter identities so that
deterministic comparison is meaningful. The single most important rule here:

    An external recruiter / staffing / ATS sending domain does NOT establish the
    identity of the hiring employer.

"robert.half@roberthalf.com writing about a Data Scientist role at Northwind" means the
employer is Northwind, not Robert Half.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# --- domains that must never be read as the hiring employer ---------------------

STAFFING_DOMAINS = frozenset(
    {
        "roberthalf.com", "randstad.com", "adecco.com", "manpower.com", "kellyservices.com",
        "hays.com", "michaelpage.com", "roberthalftechnology.com", "teksystems.com",
        "insightglobal.com", "aerotek.com", "modis.com", "apexsystems.com", "cybercoders.com",
        "motionrecruitment.com", "harnham.com", "averityhire.com", "talentburst.com",
        "collabera.com", "signifytechnology.com", "huntertechnical.com", "beaconhillstaffing.com",
        "recruiterbox.com", "staffing.com", "talentsphere.io", "hiredtalent.co",
    }
)

ATS_DOMAINS = frozenset(
    {
        "greenhouse.io", "us.greenhouse-mail.io", "greenhouse-mail.io", "lever.co",
        "hire.lever.co", "myworkday.com", "myworkdayjobs.com", "wd1.myworkdaysite.com",
        "icims.com", "taleo.net", "successfactors.com", "smartrecruiters.com",
        "jobvite.com", "workable.com", "ashbyhq.com", "bamboohr.com",
        "recruitee.com", "teamtailor.com", "breezy.hr", "pinpointhq.com",
    }
)

#: Scheduling / calendaring intermediaries. Also never the employer.
SCHEDULING_DOMAINS = frozenset(
    {
        "calendly.com", "goodtime.io", "modernloop.io", "prelude.co", "withmoxie.com",
        "hired.com", "karat.io", "karat.com", "codesignal.com", "hackerrank.com",
        "codility.com", "coderpad.io", "woven.teams", "zoom.us", "google.com",
        "microsoft.com", "calendar-notification.google.com",
    }
)

#: Consumer mail providers — a personal address proves nothing about an employer.
FREEMAIL_DOMAINS = frozenset(
    {
        "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "live.com", "icloud.com", "me.com", "aol.com", "protonmail.com", "proton.me",
    }
)

NON_EMPLOYER_DOMAINS = STAFFING_DOMAINS | ATS_DOMAINS | SCHEDULING_DOMAINS | FREEMAIL_DOMAINS

#: Corporate suffixes stripped during canonicalization.
_LEGAL_SUFFIXES = (
    # NOTE: "labs", "group" and similar are deliberately NOT stripped — they are
    # distinctive parts of brand names ("Globex Labs"), not legal suffixes.
    "incorporated", "corporation", "holdings",
    "limited", "company", "inc", "llc", "ltd", "plc", "corp", "co",
    "gmbh", "ag", "sa", "nv", "bv", "ab", "oy", "as", "pty", "pvt",
)

_PUNCT = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def canonical_company(name: str | None) -> str:
    """Lowercase, strip punctuation and legal suffixes, collapse whitespace.

    >>> canonical_company("Northwind Analytics, Inc.")
    'northwind analytics'
    >>> canonical_company("NORTHWIND ANALYTICS")
    'northwind analytics'
    """
    if not name:
        return ""
    text = _PUNCT.sub(" ", str(name).lower())
    text = _WS.sub(" ", text).strip()
    tokens = [t for t in text.split(" ") if t]
    # Strip trailing legal suffixes, but never empty the name entirely.
    while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


#: Multi-part public suffixes we need to see past to find the registrable name.
_MULTI_TLDS = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "co.in", "co.nz", "co.za",
        "com.au", "com.br", "com.mx", "com.sg", "co.kr", "com.cn", "com.tr",
    }
)


def registrable_domain(domain: str | None) -> str:
    """The registrable part of a domain, ignoring any number of subdomains.

    Real recruiting mail often arrives from a vendor subdomain of the employer —
    an external-workforce host like ``xwf.employer.com`` — so taking the first
    label would name the vendor prefix rather than the company.

    >>> registrable_domain("xwf.example.com")
    'example'
    >>> registrable_domain("talent.sub.example.co.uk")
    'example'
    """
    if not domain:
        return ""
    parts = [p for p in domain.lower().strip().lstrip("@").split(".") if p]
    if len(parts) < 2:
        return parts[0] if parts else ""
    if ".".join(parts[-2:]) in _MULTI_TLDS and len(parts) >= 3:
        return parts[-3]
    return parts[-2]


def company_from_domain(domain: str | None) -> str:
    """Derive a company guess from an email domain, or "" for non-employer domains."""
    if not domain:
        return ""
    d = domain.lower().strip().lstrip("@")
    if d in NON_EMPLOYER_DOMAINS or is_non_employer_domain(d):
        return ""
    base = registrable_domain(d)
    if not base:
        return ""
    return canonical_company(base.replace("-", " "))


def domain_of(email: str | None) -> str:
    if not email or "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip().lower()


def _domain_in(domain: str, bucket: frozenset[str]) -> bool:
    """Match a domain or any of its parent domains against a bucket."""
    if not domain:
        return False
    d = domain.lower()
    if d in bucket:
        return True
    parts = d.split(".")
    return any(".".join(parts[i:]) in bucket for i in range(1, len(parts)))


def is_staffing_domain(domain: str | None) -> bool:
    return _domain_in(domain or "", STAFFING_DOMAINS)


def is_ats_domain(domain: str | None) -> bool:
    return _domain_in(domain or "", ATS_DOMAINS)


def is_scheduling_domain(domain: str | None) -> bool:
    return _domain_in(domain or "", SCHEDULING_DOMAINS)


def is_freemail_domain(domain: str | None) -> bool:
    return _domain_in(domain or "", FREEMAIL_DOMAINS)


def is_non_employer_domain(domain: str | None) -> bool:
    """True when the sending domain cannot be taken as the hiring employer."""
    return _domain_in(domain or "", NON_EMPLOYER_DOMAINS)


# --- role titles ---------------------------------------------------------------

#: Seniority and modifier tokens removed before comparing titles.
_ROLE_NOISE = frozenset(
    {
        "senior", "sr", "junior", "jr", "staff", "principal", "lead", "head", "chief",
        "associate", "assistant", "i", "ii", "iii", "iv", "entry", "level", "mid",
        "intern", "internship", "contract", "contractor", "fulltime", "full", "time",
        "part", "remote", "hybrid", "onsite", "us", "usa", "the", "a", "an", "of", "for",
        "position", "role", "opening", "opportunity", "req", "job", "new", "grad",
        "graduate", "university", "early", "career",
    }
)

_ROLE_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "swe": "software engineer",
    "sde": "software engineer",
    "ds": "data scientist",
    "da": "data analyst",
    "mle": "machine learning engineer",
    "dev": "developer",
    "eng": "engineer",
    "engineering": "engineer",
    "scientist": "scientist",
    "analytics": "analyst",
    "bi": "business intelligence",
    "nlp": "natural language processing",
    "llm": "large language model",
}

#: Role families. Two titles in *different* families are a strong signal they are
#: different jobs — which blocks an automatic merge.
ROLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "data_analyst": ("data analyst", "business analyst", "business intelligence", "bi analyst",
                     "analytics analyst", "reporting analyst", "insights analyst"),
    "data_scientist": ("data scientist", "research scientist", "applied scientist",
                       "decision scientist", "quantitative"),
    "ml_engineer": ("machine learning engineer", "mlops", "ml platform", "ml infrastructure"),
    "ai_engineer": ("ai engineer", "artificial intelligence engineer", "genai engineer",
                    "generative ai engineer", "llm engineer", "ai software engineer"),
    "data_engineer": ("data engineer", "analytics engineer", "etl", "data platform"),
    "software_engineer": ("software engineer", "backend engineer", "frontend engineer",
                          "full stack", "fullstack", "developer", "platform engineer"),
    "product_manager": ("product manager", "product owner", "technical program manager",
                        "program manager", "project manager"),
    "designer": ("product designer", "ux designer", "ui designer", "design"),
    "devops": ("devops", "site reliability", "sre", "infrastructure engineer", "cloud engineer"),
    "research": ("research engineer", "research intern", "phd researcher"),
}


def canonical_role(title: str | None) -> str:
    """Normalize a job title: expand abbreviations, drop seniority/noise, sort-stable.

    >>> canonical_role("Senior Data Scientist II (Remote)")
    'data scientist'
    >>> canonical_role("ML Engineer")
    'machine learning engineer'
    """
    if not title:
        return ""
    text = _PUNCT.sub(" ", str(title).lower())
    text = _WS.sub(" ", text).strip()
    tokens = [t for t in text.split(" ") if t]
    expanded: list[str] = []
    for tok in tokens:
        repl = _ROLE_SYNONYMS.get(tok)
        expanded.extend(repl.split(" ") if repl else [tok])
    kept = [t for t in expanded if t not in _ROLE_NOISE and not t.isdigit()]
    # De-duplicate while preserving order (handles "engineer engineering").
    seen: set[str] = set()
    out = [t for t in kept if not (t in seen or seen.add(t))]
    return " ".join(out).strip()


def role_family(title: str | None) -> str | None:
    """Bucket a title into a coarse role family, or None when unrecognized."""
    canon = canonical_role(title)
    if not canon:
        return None
    best: tuple[int, str] | None = None
    for family, patterns in ROLE_FAMILIES.items():
        for pat in patterns:
            pat_canon = canonical_role(pat)
            if pat_canon and pat_canon in canon:
                score = len(pat_canon)
                if best is None or score > best[0]:
                    best = (score, family)
    return best[1] if best else None


def role_similarity(a: str | None, b: str | None) -> float:
    """Deterministic lexical similarity between two titles in [0, 1].

    Token overlap (Jaccard) blended with character-level ratio, so that
    "AI Engineer" vs "Artificial Intelligence Engineer" scores high while
    "Data Analyst" vs "AI Engineer" scores low.
    """
    ca, cb = canonical_role(a), canonical_role(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    ta, tb = set(ca.split()), set(cb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ratio = SequenceMatcher(None, ca, cb).ratio()
    return round(0.65 * jaccard + 0.35 * ratio, 4)


def company_similarity(a: str | None, b: str | None) -> float:
    """Lexical similarity between two company names in [0, 1]."""
    ca, cb = canonical_company(a), canonical_company(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    # A name derived from a domain has no spaces ("vantagesystems"), while the same
    # company written out does ("Vantage Systems"). Compare de-spaced forms so the
    # two reconcile.
    if ca.replace(" ", "") == cb.replace(" ", ""):
        return 0.97
    ta, tb = set(ca.split()), set(cb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        # "northwind" vs "northwind analytics" — one name contains the other.
        return 0.9
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ratio = SequenceMatcher(None, ca, cb).ratio()
    return round(0.6 * jaccard + 0.4 * ratio, 4)


# --- requisition IDs -----------------------------------------------------------

#: Requisition patterns. Deliberately anchored on an explicit label to avoid
#: harvesting random alphanumerics (tracking codes, calendar UIDs) as req IDs.
_REQ_PATTERNS = [
    # Compact, self-identifying tokens FIRST, so "REQ-4417" is captured whole rather
    # than having "REQ" consumed as a label and only "4417" kept.
    re.compile(r"\b(JR[-_]?\d{3,8}|REQ[-_]?\d{3,8}|R[-_]?\d{5,8})\b", re.IGNORECASE),
    re.compile(
        r"\b(?:requisition|req(?:uisition)?\.?|job|position|posting|vacancy|ref(?:erence)?)"
        r"[\s#:.\-]{0,4}(?:id|no|number|code)?[\s#:.\-]{0,4}"
        r"(?:is[\s:]{1,3})?"
        r"([A-Z]{2,6}[-_]?\d{3,8}|\d{4,8}[-_]?[A-Z]{0,4})\b",
        re.IGNORECASE,
    ),
]

#: Alphabetic prefixes that carry no identity (REQ-4417 and 4417 are the same req).
_REQ_PREFIXES = ("REQ", "JR", "JOB", "R")


def _req_digits(req: str) -> str:
    """Digits of a requisition ID, used to reconcile prefixed and bare forms."""
    return re.sub(r"[^0-9]", "", req or "")


def extract_requisition_id(*texts: str | None) -> str | None:
    """Pull the first plausible requisition ID out of the given texts."""
    for text in texts:
        if not text:
            continue
        for pattern in _REQ_PATTERNS:
            m = pattern.search(text)
            if m:
                return canonical_requisition(m.group(1))
    return None


def canonical_requisition(req: str | None) -> str:
    """Uppercase and strip separators so REQ-4417 == req_4417 == REQ4417."""
    if not req:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(req).upper())


def requisitions_conflict(a: str | None, b: str | None) -> bool:
    """True only when both requisition IDs are known AND genuinely different.

    "REQ-4417", "req_4417" and "4417" all refer to the same requisition, so they do
    not conflict. "REQ-4417" vs "REQ-5120" does — and that is a hard merge blocker.
    """
    ca, cb = canonical_requisition(a), canonical_requisition(b)
    if not ca or not cb or ca == cb:
        return False
    da, db = _req_digits(ca), _req_digits(cb)
    if da and db and da == db:
        # Same number; differ only by a non-identifying prefix.
        stripped = {ca.replace(da, ""), cb.replace(db, "")}
        if all(s == "" or s in _REQ_PREFIXES for s in stripped):
            return False
    return True


# --- recruiter identity --------------------------------------------------------

_HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "dr", "prof"})


def canonical_person(name: str | None) -> str:
    """Normalize a person name for comparison."""
    if not name:
        return ""
    text = _PUNCT.sub(" ", str(name).lower())
    tokens = [t for t in _WS.sub(" ", text).strip().split(" ") if t and t not in _HONORIFICS]
    return " ".join(tokens)


def name_from_email(email: str | None) -> str:
    """Best-effort human name from a local-part like `sarah.chen` -> `sarah chen`."""
    if not email or "@" not in email:
        return ""
    local = email.split("@", 1)[0]
    local = re.sub(r"\d+", " ", local)
    local = re.sub(r"[._\-+]+", " ", local)
    local = _WS.sub(" ", local).strip()
    if local.lower() in {"no reply", "noreply", "do not reply", "donotreply", "info",
                         "careers", "jobs", "recruiting", "talent", "hr", "hello", "team",
                         "notifications", "support", "mailer daemon"}:
        return ""
    return canonical_person(local)
