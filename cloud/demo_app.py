"""Public fictional demo: live reconstruction with a separate SQLite DB per session.

Deploy this file on Streamlit Community Cloud. Gmail is never connected here.
"""
from pathlib import Path
import os
import sys

# Configure before importing the application or loading settings.
os.environ['EMBEDDINGS_ENABLED'] = 'false'
os.environ['LLM_REVIEW_ENABLED'] = 'false'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import app as ui
from interviewtrack.db import Database
from interviewtrack.embeddings import get_backend
from interviewtrack.fixtures.mailboxes import PROFILE_REVIEWER
from interviewtrack.interview_policy import interview_bucket
from interviewtrack.service import InterviewTrackService, search_applications, sort_applications


def landing():
    st.title('📬 InterviewTrack AI')
    st.caption('Keep screening invitations and interview next steps from getting buried in your inbox.')
    demo, gmail = st.columns(2, gap='large')
    with demo:
        st.subheader('Demo')
        st.write('Explore fictional interview journeys, multiple rounds, staffing handoffs, offers and rejections.')
        st.caption('No Gmail account or credentials needed.')
        if st.button('Start Demo', type='primary', use_container_width=True):
            st.session_state['_demo_started'] = True
            st.rerun()
    with gmail:
        st.subheader('Gmail Access')
        st.info('Gmail testing is available to approved Google OAuth test users.')
        st.write('Ask the administrator to add your Gmail address as a test user in the Google Cloud OAuth consent configuration.')
        st.caption('After approval, follow the Gmail setup instructions to authorize your account locally. This public app does not connect to real Gmail accounts.')
        st.link_button('View Gmail setup', 'https://github.com/madhumitha-gv/InterviewTrackAI#gmail-mode-setup', use_container_width=True)


def main():
    if not st.session_state.get('_demo_started'):
        landing()
        return
    if '_hosted_service' not in st.session_state:
        # Do not cache a mutable database across visitors or use the bundled DB.
        svc = InterviewTrackService(db=Database(':memory:'), backend=get_backend(force_hashed=True))
        outcome = svc.load_demo_data(profile_key='reviewer')
        if outcome.errors:
            svc.db.close()
            st.error('The fictional demo could not initialize. Please reload the page.')
            st.stop()
        st.session_state['_hosted_service'] = svc
        st.session_state.update(user_id='demo:reviewer', user_email=PROFILE_REVIEWER.user_email,
            user_name='Avery Quinn (fictional)', mode='demo', demo_profile_key='reviewer')
    svc = st.session_state['_hosted_service']
    uid = st.session_state['user_id']
    with st.sidebar:
        st.markdown('**InterviewTrack AI**')
        st.caption('Public demo · fictional emails')
        st.caption('Explore screening, multiple rounds, staffing handoffs, offers and rejections.')
        if st.button('Refresh demo', use_container_width=True):
            svc.load_demo_data(profile_key='reviewer')
            st.rerun()
        if st.button('Reset demo', use_container_width=True):
            svc.db.close()
            st.session_state.clear()
            st.session_state['_demo_started'] = True
            st.rerun()
        if st.button('Back to start', use_container_width=True):
            svc.db.close()
            st.session_state.clear()
            st.rerun()
        st.link_button('View source on GitHub', 'https://github.com/madhumitha-gv/InterviewTrackAI', use_container_width=True)
        st.caption('Changes apply only to this browser session. No Gmail connection or credentials are used.')
    st.title('📬 InterviewTrack AI')
    st.caption('Keep screening invitations and interview next steps from getting buried in your inbox.')
    st.info('Fictional demo: start with Lumen Robotics — Machine Learning Engineer for the complete screening-to-offer journey.')
    apps = svc.interview_applications(uid, PROFILE_REVIEWER.user_email)
    top = st.columns([3, 2, 1])
    query = top[0].text_input('Search by company, position or recruiter', placeholder='e.g. Lumen Robotics')
    selected = top[1].radio('Filter', ['Active', 'Rejected'], horizontal=True)
    expand_all = top[2].toggle('Expand all', value=False)
    ui.render_review_queue(svc, apps)
    visible = sort_applications([a for a in search_applications(apps, query) if interview_bucket(a) == selected])
    st.caption(f'{len(visible)} {selected.lower()} application(s)')
    if not visible:
        st.info('No applications match your search.')
    for application in visible:
        ui.render_card(svc, application, apps, expanded=expand_all)


if __name__ == '__main__':
    main()
