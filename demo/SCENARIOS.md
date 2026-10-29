# Fictional inbox scenarios

All 31 emails are invented. No real mailbox or authorized Gmail account is included.

| Employer / role | Journey | Dashboard |
| --- | --- | --- |
| Juniper Labs / AI Engineer | Screening invitation and candidate clarification | Active |
| Harbor Energy / Data Engineer | Screening, technical interview, assignment, submission, final interview, follow-up, offer | Active / Offer |
| Harbor Energy / Platform Engineer | Screening, rejection | Rejected |
| Lumen Robotics / Machine Learning Engineer | Screening invitation, screening completion/next step, technical rounds 1 and 2, assignment, submission, final interview, follow-up, offer | Active / Offer |
| Lumen Robotics / Backend Engineer | Screening, technical interview, rejection | Rejected |
| Summit Analytics / Data Engineer | Screening, technical rounds 1 and 2, candidate follow-up | Active |

The Lumen Machine Learning Engineer journey is the complete staffing-assisted example. Elena Ward, Sasha Reed and Devon Park coordinate for Clearpath Staffing. Sasha and Devon are CC recipients; messages change senders and threads, and the employer sends the final invitation and offer. These contacts also appear on a different Lumen role and a Summit role. Shared contacts must not collapse separate roles or employers.

Three job-alert/advertisement examples are excluded. An application receipt is stored as internal context but is excluded from interview cards. Expected dashboard totals: 4 Active (including 2 Offers), 2 Rejected.

Open `fictional-emails.json` to inspect every email, including From, To, CC, body, date and synthetic thread IDs. `fictional-inbox.db` uses the application SQLite schema and stores the reconstruction output. In the app, select “Fictional inbox — screening through offer” and Start Demo Mode; existing sessions can click Refresh demo. Expand a timeline and View email to inspect the full text.

The input contains 31 messages. The database stores 29 after retrieval filtering; later classification and the interview gate remove remaining noise/context from the six dashboard cards. The JSON export retains all 31 inputs for inspection.
