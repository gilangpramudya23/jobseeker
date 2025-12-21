import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import agents with error handling
try:
    from src.agents.orchestrator import Orchestrator
    from src.agents.advisor_agent import AdvisorAgent
    from src.agents.cover_letter_agent import CoverLetterAgent
    from src.agents.interview_agent import InterviewAgent
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    st.warning("streamlit_mic_recorder not installed. Mock Interview feature will be limited.")
    mic_recorder = None

# Load environment variables
load_dotenv()

# Konfigurasi Halaman
st.set_page_config(page_title="AI Career Hub", layout="wide")

# Inisialisasi Agent (menggunakan cache agar tidak reload setiap saat)
@st.cache_resource
def init_agents():
    """Initialize all agents with error handling"""
    try:
        return {
            "orchestrator": Orchestrator(),
            "advisor": AdvisorAgent(),
            "cover_letter": CoverLetterAgent(),
            "interview": InterviewAgent()
        }
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        return None

agents = init_agents()

if agents is None:
    st.error("Failed to initialize agents. Please check your configuration.")
    st.stop()

# Sidebar Navigasi
st.sidebar.title("🚀 Career AI Agent")
menu = st.sidebar.radio("Pilih Fitur:", [
    "Smart Chat (SQL & RAG)", 
    "Career Advisor & CV Analysis", 
    "Cover Letter Generator", 
    "Mock Interview (Voice)"
])

st.sidebar.divider()
st.sidebar.info("Gunakan sidebar untuk berpindah antar fungsi agent.")

# --- 1. SMART CHAT (ORCHESTRATOR) ---
if menu == "Smart Chat (SQL & RAG)":
    st.header("💬 Smart Career Chat")
    st.write("Tanyakan data statistik (SQL) atau informasi deskriptif lowongan (RAG).")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Contoh: Berapa jumlah lowongan Python? atau Apa syarat Software Engineer?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                try:
                    response = agents["orchestrator"].route_query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- 2. CAREER ADVISOR ---
elif menu == "Career Advisor & CV Analysis":
    st.header("👨‍💼 Career Consultant")
    uploaded_file = st.file_uploader("Upload CV kamu (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Simpan file sementara
        with open("temp_cv.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analisis CV & Cari Lowongan"):
            with st.spinner("Menganalisis profil kamu..."):
                try:
                    report = agents["advisor"].analyze_and_recommend("temp_cv.pdf")
                    st.markdown("### Laporan Konsultasi")
                    st.write(report)
                except Exception as e:
                    st.error(f"Error analyzing CV: {str(e)}")
                finally:
                    if os.path.exists("temp_cv.pdf"):
                        os.remove("temp_cv.pdf")

# --- 3. COVER LETTER GENERATOR ---
elif menu == "Cover Letter Generator":
    st.header("📝 Tailored Cover Letter")
    col1, col2 = st.columns(2)
    
    with col1:
        cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cl_cv")
    with col2:
        job_desc = st.text_area("Tempel Deskripsi Pekerjaan di sini:", height=200)

    if st.button("Generate Cover Letter"):
        if cv_file and job_desc:
            with open("temp_cl_cv.pdf", "wb") as f:
                f.write(cv_file.getbuffer())
            
            with st.spinner("Menulis Cover Letter..."):
                try:
                    letter = agents["cover_letter"].generate_cover_letter("temp_cl_cv.pdf", job_desc)
                    st.subheader("Hasil Cover Letter:")
                    st.text_area("Salin hasil di sini:", value=letter, height=400)
                except Exception as e:
                    st.error(f"Error generating cover letter: {str(e)}")
                finally:
                    if os.path.exists("temp_cl_cv.pdf"):
                        os.remove("temp_cl_cv.pdf")
        else:
            st.warning("Mohon upload CV dan isi deskripsi pekerjaan.")

# --- 4. MOCK INTERVIEW (VOICE) ---
elif menu == "Mock Interview (Voice)":
    st.header("🎤 AI Mock Interview")
    
    if mic_recorder is None:
        st.error("Mock Interview feature requires streamlit_mic_recorder. Please install it: pip install streamlit-mic-recorder")
        st.stop()
    
    # Inisialisasi state jika belum ada
    if "int_history" not in st.session_state:
        st.session_state.int_history = ""
        st.session_state.last_q = "Halo! Bisa ceritakan tentang diri Anda?"
        # Add initial greeting to history
        st.session_state.int_history = f"Interviewer: {st.session_state.last_q}\n"

    # Display the last question
    st.chat_message("assistant").write(st.session_state.last_q)

    # Rekam Suara
    audio = mic_recorder(
        start_prompt="Klik untuk Bicara 🎙️",
        stop_prompt="Berhenti & Kirim 📤",
        key='recorder'
    )

    if audio:
        with st.spinner("AI sedang mendengarkan..."):
            try:
                # 1. Konversi Audio ke Teks menggunakan OpenAI Whisper
                from openai import OpenAI
                client = OpenAI()
                
                # Simpan buffer audio sementara
                with open("temp_audio.mp3", "wb") as f:
                    f.write(audio['bytes'])
                
                # Whisper API
                with open("temp_audio.mp3", "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                user_text = transcript.text
                
                st.chat_message("user").write(user_text)

                # 2. Kirim teks ke Interview Agent
                response = agents["interview"].get_response(
                    st.session_state.int_history, 
                    user_text
                )
                
                # 3. Update State
                st.session_state.int_history += f"Candidate: {user_text}\n"
                st.session_state.int_history += f"Interviewer: {response}\n"
                st.session_state.last_q = response
                
                # 4. Clean up temp file
                if os.path.exists("temp_audio.mp3"):
                    os.remove("temp_audio.mp3")
                
                # 5. Rerun to show new question
                st.rerun()
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
                # Clean up on error
                if os.path.exists("temp_audio.mp3"):
                    os.remove("temp_audio.mp3")

    if st.button("Reset Interview"):
        # Clear all interview state
        if "int_history" in st.session_state:
            del st.session_state.int_history
        if "last_q" in st.session_state:
            del st.session_state.last_q
        st.rerun()
