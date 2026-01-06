import streamlit as st
import os
import requests
import base64
import json
import hashlib
import re
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import openai

# Load environment variables
load_dotenv()

# Cloud Run Configuration
BASE_URL = "https://aicareerhub3img-482891049098.asia-southeast2.run.app"

# Page Configuration
st.set_page_config(page_title="AI Career Hub | Cloud", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Premium Look (Mirrored from app.py)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    /* Global Base */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Titles & Headers */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
    }

    .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        text-align: center;
        width: 100%;
        display: block;
    }

    /* Glass Containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin: 1rem 0;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    }

    /* Form Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 0px;
        font-weight: 600;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- API HELPER FUNCTIONS ---

def api_chat(message):
    try:
        response = requests.post(f"{BASE_URL}/chat", json={"message": message})
        response.raise_for_status()
        return response.json().get("response", "No response from AI.")
    except Exception as e:
        return f"Error: {str(e)}"

def api_analyze_cv(cv_base64):
    try:
        response = requests.post(f"{BASE_URL}/cv/analyze", json={"cv_base64": cv_base64}, timeout=150)
        response.raise_for_status()
        return response.json().get("analysis", "Analysis failed.")
    except Exception as e:
        return f"Error: {str(e)}"

def api_generate_cover_letter(cv_base64, job_desc):
    try:
        response = requests.post(
            f"{BASE_URL}/cover-letter/generate", 
            json={"cv_base64": cv_base64, "job_description": job_desc},
            timeout=150
        )
        response.raise_for_status()
        return response.json().get("cover_letter", "Generation failed.")
    except Exception as e:
        return f"Error: {str(e)}"

def api_start_interview():
    try:
        response = requests.get(f"{BASE_URL}/interview/start")
        response.raise_for_status()
        return response.json().get("first_question", "Hello! Could you introduce yourself?")
    except Exception as e:
        return "Hello! Let's start the interview. Can you tell me about yourself?"

def api_interview_chat(answer, history, job_desc="", cv_text=""):
    try:
        payload = {
            "candidate_answer": answer,
            "conversation_history": history,
            "job_description": job_desc,
            "cv_text": cv_text
        }
        response = requests.post(f"{BASE_URL}/interview/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("interviewer_response", "Sorry, I missed that. Could you repeat?")
    except Exception as e:
        return f"Error: {str(e)}"

# --- UI LOGIC ---

# Initialize Session State
if "track" not in st.session_state:
    st.session_state.track = "🚀 Career Co-Pilot"
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "cv_base64" not in st.session_state:
    st.session_state.cv_base64 = None
if "cv_text" not in st.session_state:
    st.session_state.cv_text = None  
if "advisor_report" not in st.session_state:
    st.session_state.advisor_report = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_job_desc" not in st.session_state:
    st.session_state.selected_job_desc = None

def reset_session():
    # Only clear actual session data, keep persistent track if needed
    keys_to_keep = ["track"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.page = "landing"
    st.rerun()

# Sidebar Navigation
with st.sidebar:
    st.title("🛡️ AI Career Hub")
    st.caption("Cloud Deployment Version")
    st.markdown("---")
    selected_track = st.radio("Navigation", ["🚀 Career Co-Pilot", "💬 Smart Chat", "ℹ️ About"])
    
    if selected_track != st.session_state.track:
        st.session_state.track = selected_track
        st.rerun()
        
    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()

# --- TRACK 1: CAREER CO-PILOT ---
if st.session_state.track == "🚀 Career Co-Pilot":
    if st.session_state.page == "landing":
        st.markdown('<div style="text-align: center; padding: 5rem 0 2rem 0;">', unsafe_allow_html=True)
        st.markdown('<h1 class="gradient-text">Unlock Your Future</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.25rem; color: #94a3b8; max-width: 700px; margin: 0 auto 3rem auto; text-align: center; line-height: 1.6;">Upload your CV and let our Cloud-powered AI agents guide you. We analyze, recommend, and prepare you for success.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("Drop your CV here (PDF)", type=["pdf"], label_visibility="collapsed")
            if uploaded_file:
                if st.button("Analyze My Career 🚀", use_container_width=True):
                    with st.status("Connecting to Cloud AI...", expanded=True) as status:
                        st.write("Encoding document...")
                        cv_bytes = uploaded_file.read()
                        cv_base64 = base64.b64encode(cv_bytes).decode('utf-8')
                        st.session_state.cv_base64 = cv_base64
                        
                        st.write("Running Deep Career Analysis (may take 1-2 mins on first run)...")
                        report = api_analyze_cv(cv_base64)
                        st.session_state.advisor_report = report
                        
                        # Use part of report as context for other agents
                        st.session_state.cv_text = report[:3000]
                        
                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                    
                    st.session_state.page = "dashboard"
                    st.rerun()

    elif st.session_state.page == "dashboard":
        st.header("🎯 Your Career Roadmap")
        
        if not st.session_state.advisor_report:
            st.warning("No analysis found. Please upload your CV.")
            if st.button("Back to Upload"):
                st.session_state.page = "landing"
                st.rerun()
        else:
            st.markdown(f'<div class="glass-container">{st.session_state.advisor_report}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Action Center")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 16px; border-left: 4px solid #6366f1;">
                    <h4 style="margin: 0;">Practice & Prepare</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem;">Input a specific job to get a tailored cover letter and interview practice.</p>
                </div>
                """, unsafe_allow_html=True)
                job_title = st.text_input("Target Job Title", placeholder="e.g. Senior Backend Engineer")
                job_desc = st.text_area("Job Description", placeholder="Paste the job requirements here...", height=150)
                if st.button("Open Workspace 🚀", use_container_width=True):
                    if job_desc:
                        st.session_state.selected_job_desc = job_desc
                        st.session_state.selected_job_title = job_title or "Target Role"
                        st.session_state.page = "workspace"
                        st.rerun()
                    else:
                        st.error("Please provide a job description.")
            with col2:
                st.markdown("""
                <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 16px; border-left: 4px solid #a855f7;">
                    <h4 style="margin: 0;">Market Insights</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem;">Ask our AI about current trends, salaries, or companies matching your profile.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Consult AI Assistant 💬", use_container_width=True):
                    st.session_state.track = "💬 Smart Chat"
                    st.rerun()

    elif st.session_state.page == "workspace":
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Back"):
                st.session_state.page = "dashboard"
                st.rerun()
        
        with col_title:
            st.markdown(f"## 🛠️ Workspace: <span style='color: #818cf8;'>{st.session_state.get('selected_job_title', 'Target Role')}</span>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📝 Cover Letter", "🎙️ Interview Sim"])
        
        with tab1:
            st.subheader("Tailored Cover Letter")
            if st.button("Generate Winner Letter"):
                with st.spinner("Our agent is crafting your letter..."):
                    letter = api_generate_cover_letter(st.session_state.cv_base64, st.session_state.selected_job_desc)
                    st.markdown("### Preview")
                    st.markdown(f'<div style="background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 12px; white-space: pre-wrap;">{letter}</div>', unsafe_allow_html=True)
                    st.download_button("Download as Text", letter, file_name="cover_letter.txt")
            else:
                st.info("Let the AI write a cover letter that highlights your unique strengths for this role.")

        with tab2:
            st.subheader("Interactive Interview Practice")
            
            if "interview_log" not in st.session_state:
                st.session_state.interview_log = [{"role": "assistant", "content": api_start_interview()}]
                st.session_state.int_history_text = ""
                st.session_state.int_ended = False

            col_chat, col_ctrl = st.columns([3, 1])
            with col_ctrl:
                st.markdown('<div class="glass-container" style="padding: 1rem;">', unsafe_allow_html=True)
                st.write("**Controls**")
                if st.button("🔄 Reset", use_container_width=True):
                    if "interview_log" in st.session_state: del st.session_state.interview_log
                    st.rerun()
                if st.button("🏁 End Session", use_container_width=True):
                    st.session_state.int_ended = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with col_chat:
                if not st.session_state.int_ended:
                    current_q = st.session_state.interview_log[-1]["content"]
                    st.markdown(f"**Interviewer:** {current_q}")
                    
                    audio_data = mic_recorder(start_prompt="Speak your answer 🎤", stop_prompt="Stop & Send ✅", key='cloud_mic')
                    if audio_data:
                        audio_bytes = audio_data['bytes']
                        audio_hash = hashlib.md5(audio_bytes).hexdigest()
                        
                        if "last_processed_audio_hash" not in st.session_state or st.session_state.last_processed_audio_hash != audio_hash:
                            with st.status("Thinking...", expanded=False):
                                # Local Whisper Transcription
                                with open("temp_audio.mp3", "wb") as f: f.write(audio_bytes)
                                client = openai.OpenAI()
                                with open("temp_audio.mp3", "rb") as af:
                                    transcript = client.audio.transcriptions.create(model="whisper-1", file=af)
                                user_text = transcript.text
                                os.remove("temp_audio.mp3")
                                
                                st.session_state.interview_log.append({"role": "user", "content": user_text})
                                
                                # Call API for interviewer response
                                next_q = api_interview_chat(
                                    user_text, 
                                    st.session_state.int_history_text,
                                    st.session_state.selected_job_desc,
                                    st.session_state.cv_text
                                )
                                st.session_state.interview_log.append({"role": "assistant", "content": next_q})
                                st.session_state.int_history_text += f"Q: {current_q}\nA: {user_text}\n"
                                
                                st.session_state.last_processed_audio_hash = audio_hash
                                st.rerun()
                else:
                    st.success("🎉 Session Completed! You can view your history below.")
                    if st.button("New Practice Session"):
                        st.session_state.int_ended = False
                        if "interview_log" in st.session_state: del st.session_state.interview_log
                        st.rerun()

                st.markdown("---")
                for msg in reversed(st.session_state.interview_log):
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

# --- TRACK 2: SMART CHAT ---
elif st.session_state.track == "💬 Smart Chat":
    st.header("💬 AI Career Assistant")
    st.write("Ask our Orchestrator about job markets, SQL analytics, or expert advice.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Processing your query via Cloud AI..."):
                response = api_chat(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- TRACK 3: ABOUT ---
elif st.session_state.track == "ℹ️ About":
    st.markdown('<div class="glass-container" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">Cloud Deployment Edition</h1>', unsafe_allow_html=True)
    st.markdown("""
        This version of **AI Career Hub** is optimized for frontend-only deployment.
        It connects to a robust Google Cloud Run backend to perform heavy-duty AI tasks.
    """)
    st.markdown("---")
    st.markdown("### 🛠️ Architecture")
    st.markdown("""
    - **Frontend:** Streamlit Hosted
    - **Backend:** FastAPI on Google Cloud Run
    - **Database:** Qdrant (Vector) & SQLite
    - **Intelligence:** GPT-4o-mini & Whisper
    """)
    st.markdown('</div>', unsafe_allow_html=True)