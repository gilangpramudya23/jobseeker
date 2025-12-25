import streamlit as st
import os
from dotenv import load_dotenv
from src.agents.orchestrator import Orchestrator
from src.agents.advisor_agent import AdvisorAgent
from src.agents.cover_letter_agent import CoverLetterAgent
from src.agents.interview_agent import InterviewAgent
from streamlit_mic_recorder import mic_recorder

# Load environment variables
load_dotenv()

# Konfigurasi Halaman
st.set_page_config(page_title="AI Career Hub", layout="wide")

# Inisialisasi Agent (menggunakan cache agar tidak reload setiap saat)
@st.cache_resource
def init_agents():
    return {
        "orchestrator": Orchestrator(),
        "advisor": AdvisorAgent(),
        "cover_letter": CoverLetterAgent(),
        "interview": InterviewAgent()
    }

agents = init_agents()

# Sidebar Navigasi
st.sidebar.title("🚀 Career AI Agent")
menu = st.sidebar.radio("Pilih Fitur:", [
    "Smart Chat", 
    "Career Advisor & CV Analysis", 
    "Cover Letter Generator", 
    "AI Interview Assistant (Voice)"
])

st.sidebar.divider()
st.sidebar.info("Gunakan sidebar untuk berpindah antar fungsi agent.")

# --- 1. SMART CHAT (ORCHESTRATOR) ---
if menu == "Smart Chat":
    st.header("💬 Smart Career Chat")
    st.write("Tanyakan data statistik atau informasi deskriptif lowongan")
    
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
                response = agents["orchestrator"].route_query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add clear chat button
    if len(st.session_state.messages) > 0:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
                
# --- 2. CAREER ADVISOR ---

elif menu == "Career Advisor & CV Analysis":
    st.header("👨‍💼 Career Consultant")

    # 1. Inisialisasi Session State untuk chat Career Advisor
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []

    uploaded_file = st.file_uploader("Upload CV kamu (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Simpan file sementara
        with open("temp_cv.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analisis CV & Cari Lowongan"):
            with st.spinner("Menganalisis profil kamu..."):
                report = agents["advisor"].analyze_and_recommend("temp_cv.pdf")
                st.session_state.current_report = report # Simpan report di state
                
                # Masukkan hasil laporan ke dalam history chat sebagai pesan awal AI
                st.session_state.advisor_messages.append({"role": "assistant", "content": report})
            os.remove("temp_cv.pdf")

    # 2. Tampilkan Riwayat Chat (jika sudah ada analisis)
    
    with st.expander("ℹ️ Tips untuk hasil terbaik"):
        st.markdown("""
        **Gunakan PDF berbentuk teks untuk mendapatkan report yang lebih akurat.**
        
        **Format yang didukung:**
        - ✅ PDF dengan teks (Berbagai bahasa)
        """)
            
    for message in st.session_state.advisor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Input Message untuk Chat (Hanya muncul jika sudah ada analisis awal)
    if st.session_state.advisor_messages:
        if prompt := st.chat_input("Tanyakan lebih detail tentang saran karirmu..."):
            # Tampilkan pesan user
            st.session_state.advisor_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Minta respon dari Agent (Gunakan Orchestrator atau Advisor)
            with st.chat_message("assistant"):
                with st.spinner("Berpikir..."):
                    # Kita buat history string dari advisor_messages untuk konteks
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.advisor_messages[-5:]])
                    
                    # Kamu bisa memanggil orchestrator agar AI tetap ingat konteks CV-mu
                    response = agents["orchestrator"].route_request(prompt, history_text)
                    st.markdown(response)
            
            # Simpan respon AI
            st.session_state.advisor_messages.append({"role": "assistant", "content": response})
            
    
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
                letter = agents["cover_letter"].generate_cover_letter("temp_cl_cv.pdf", job_desc)
                st.subheader("Hasil Cover Letter:")
                st.text_area("Salin hasil di sini:", value=letter, height=400)
            os.remove("temp_cl_cv.pdf")
        else:
            st.warning("Mohon upload CV dan isi deskripsi pekerjaan.")

# --- 4. MOCK INTERVIEW (VOICE) ---
# Di dalam app.py pada bagian menu "Mock Interview"

from streamlit_mic_recorder import mic_recorder
import openai
import os

# --- DI DALAM KONDISI MENU INTERVIEW ---
if menu == "AI Interview Assistant (Voice)":
    # Header dengan gaya Dashboard
    st.title("🎙️ AI Career Coach: Interview Room")
    st.caption("Berlatihlah bicara secara alami. Jawaban Anda akan ditranskripsi dan dianalisis secara otomatis.")
    st.divider()

    # Layout Kolom: Kiri untuk Chat, Kanan untuk Instruksi/Tips
    col_main, col_sidebar = st.columns([2, 1])

    with col_sidebar:
        st.subheader("💡 Tips Interview")
        st.info("""
        - **Kontak Mata:** Meskipun virtual, tetap fokus pada kamera.
        - **Metode STAR:** Gunakan (Situation, Task, Action, Result) untuk jawaban teknis.
        - **Suara Jelas:** Bicara dengan tempo yang tenang.
        """)
        
        if st.button("🔄 Reset Sesi Interview"):
            # Logika reset state jika dibutuhkan
            st.session_state.interview_log = []
            st.session_state.interview_history = "AI Interviewer: Hello! Let's start. Tell me about yourself.\n"
            st.session_state.current_q = "Hello! Let's start. Tell me about yourself."
            st.rerun()

    with col_main:
        # 1. Area Pertanyaan Aktif (Dibuat menonjol)
        st.markdown("### 🤖 Pertanyaan Saat Ini:")
        with st.container(border=True):
            st.subheader(st.session_state.current_q)
            st.write("---")
            # Widget Mic ditempatkan tepat di bawah pertanyaan
            st.write("Klik tombol di bawah untuk merekam jawaban Anda:")
            audio_data = mic_recorder(
                start_prompt="Mulai Bicara 🎤",
                stop_prompt="Selesai & Kirim ✅",
                key='interview_mic_unique' 
            )

        # 2. Riwayat Percakapan (Menggunakan st.chat_message agar unik)
        st.markdown("### 📝 Riwayat Jawaban Anda")
        if "interview_log" not in st.session_state:
            st.session_state.interview_log = []

        if not st.session_state.interview_log:
            st.info("Belum ada jawaban yang terekam.")
        else:
            # Tampilkan riwayat dengan gaya chat
            for i, msg in enumerate(st.session_state.interview_log):
                with st.chat_message("user"):
                    st.write(msg)

    # --- LOGIKA FUNGSIONAL (TIDAK BERUBAH) ---
    if "interview_history" not in st.session_state:
        st.session_state.interview_history = "AI Interviewer: Hello! Let's start. Tell me about yourself.\n"
        st.session_state.current_q = "Hello! Let's start. Tell me about yourself."

    if audio_data:
        audio_bytes = audio_data['bytes']
        
        if "last_processed_audio" not in st.session_state or st.session_state.last_processed_audio != audio_bytes:
            with st.status("Sedang memproses suara Anda...", expanded=True) as status:
                st.write("Mentranskripsi audio (Whisper)...")
                with open("temp_interview.mp3", "wb") as f:
                    f.write(audio_bytes)
                
                client = openai.OpenAI()
                with open("temp_interview.mp3", "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                user_text = transcript.text
                st.session_state.interview_log.append(user_text)

                st.write("Menganalisis jawaban & menyiapkan pertanyaan baru...")
                response = agents["interview"].get_response(
                    st.session_state.interview_history, 
                    user_text
                )
                
                st.session_state.interview_history += f"Candidate: {user_text}\nInterviewer: {response}\n"
                st.session_state.current_q = response
                st.session_state.last_processed_audio = audio_bytes
                
                os.remove("temp_interview.mp3")
                status.update(label="Proses selesai!", state="complete", expanded=False)
            
            st.rerun()
