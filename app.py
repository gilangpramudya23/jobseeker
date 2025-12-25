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

import pytesseract
from pdf2image import convert_from_path

print("OCR is ready!")

elif menu == "Career Advisor & CV Analysis":
    st.header("👨‍💼 Career Consultant")
    st.caption("💡 Supports both text-based and scanned/image PDFs!")
    
    # Check OCR availability
    try:
        import pytesseract
        from pdf2image import convert_from_path
        ocr_status = "✅ OCR Available - Can process scanned documents"
        ocr_color = "success"
    except ImportError:
        ocr_status = "⚠️ OCR Not Available - Only text-based PDFs supported. Install: pip install pdf2image pytesseract pillow"
        ocr_color = "warning"
    
    st.info(ocr_status)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CV kamu (PDF)", 
        type=["pdf"],
        help="Supports both regular PDFs and scanned documents"
    )
    
    if uploaded_file:
        # Display file info
        file_size = uploaded_file.size / 1024  # KB
        st.write(f"📄 File: **{uploaded_file.name}** ({file_size:.1f} KB)")
        
        # Simpan file sementara
        temp_path = "temp_cv.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🔍 Analisis CV & Cari Lowongan", type="primary"):
            with st.spinner("Mengekstrak teks dari CV..."):
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📖 Membaca file PDF...")
                    progress_bar.progress(20)
                    
                    status_text.text("🔍 Menganalisis profil kandidat...")
                    progress_bar.progress(40)
                    
                    # Analyze CV
                    report = agents["advisor"].analyze_and_recommend(temp_path)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analisis selesai!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📋 Laporan Konsultasi Karir")
                    
                    # Check if it's an error message
                    if report.startswith("Error:"):
                        st.error(report)
                    else:
                        st.markdown(report)
                        
                        # Add download button for report
                        st.download_button(
                            label="📥 Download Laporan",
                            data=report,
                            file_name="career_consultation_report.txt",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis CV: {str(e)}")
                    st.info("Jika CV Anda berupa scan/gambar, pastikan OCR libraries sudah terinstall.")
                
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    progress_bar.empty()
                    status_text.empty()
    
    # Help section
    with st.expander("ℹ️ Tips untuk hasil terbaik"):
        st.markdown("""
        **Untuk CV berbasis teks:**
        - Format PDF standar akan diproses dengan cepat
        
        **Untuk CV scan/gambar:**
        - Gunakan resolusi tinggi (300 DPI atau lebih)
        - Pastikan teks terlihat jelas dan tidak buram
        - Hindari background yang terlalu gelap
        - OCR libraries harus terinstall (lihat dokumentasi)
        
        **Format yang didukung:**
        - ✅ PDF dengan teks (native)
        - ✅ PDF hasil scan
        - ✅ PDF berisi gambar CV
        """)

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

# --- DI DALAM KONDISI MENU INTERVIEW ---
if menu == "Mock Interview (Voice)":
    st.header("🎤 AI Mock Interview")

    if "interview_log" not in st.session_state:
        st.session_state.interview_log = []

    for msg in st.session_state.interview_log[-3:]:
        with st.success(f"**You:** {msg}"):
            st.write(msg)
    
    # 1. Inisialisasi State (Hanya jalan sekali di awal)
    if "interview_history" not in st.session_state:
        st.session_state.interview_history = "AI Interviewer: Hello! Let's start. Tell me about yourself.\n"
        st.session_state.current_q = "Hello! Let's start. Tell me about yourself."
    
    # 2. Tampilkan Pertanyaan AI
    st.info(f"**AI Interviewer:** {st.session_state.current_q}")

    # 3. Widget Mic
    audio_data = mic_recorder(
        start_prompt="Mulai Bicara 🎙️",
        stop_prompt="Kirim Jawaban ✅",
        key='interview_mic_unique' 
    )

    # 4. Logika Pemrosesan (Taruh tepat di bawah widget mic)
    if audio_data:
        audio_bytes = audio_data['bytes']
        
        # Cek apakah audio ini baru atau duplikat dari rerun sebelumnya
        if "last_processed_audio" not in st.session_state or st.session_state.last_processed_audio != audio_bytes:
            
            # --- PROSES MULAI DI SINI ---
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
            
            # Panggil agent untuk jawaban
            response = agents["interview"].get_response(
                st.session_state.interview_history, 
                user_text
            )
            
            
            # Simpan ke history dan tandai audio sudah diproses
            st.session_state.interview_history += f"Candidate: {user_text}\nInterviewer: {response}\n"
            st.session_state.current_q = response
            st.session_state.last_processed_audio = audio_bytes # KUNCI PENTING
            
            os.remove("temp_interview.mp3")
            st.rerun() # Refresh tampilan untuk memunculkan pertanyaan baru
            st.success(f"You {user_text}")











































