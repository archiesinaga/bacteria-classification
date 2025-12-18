import streamlit as st
from huggingface_hub import hf_hub_download
import os
from models.loader import ModelHandler
from prediksi import BacteriaClassificationApp
from evaluasi import ModelEvaluation

# ================== KONFIGURASI ==================
st.set_page_config(
    page_title="Klasifikasi Bakteri",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi Model
HF_REPO_ID = "archiesinaga/bacteria-classification-convnext"  # ← GANTI dengan repo Hugging Face Anda!
MODEL_FILENAME = "best_model.keras"
LOCAL_MODEL_PATH = "best_model.keras"

# Daftar 33 kelas bakteri
CLASS_NAMES = [
    'Acinetobacter baumannii', 
    'Actinomyces israelii',
    'Bacteroides fragilis', 
    'Bifidobacterium spp',
    'Candida albicans',
    'Clostridium perfringens', 
    'Enterococcus faecalis',
    'Enterococcus faecium',
    'Escherichia coli', 
    'Fusobacterium',
    'Lactobacillus casei', 
    'Lactobacillus crispatus',
    'Lactobacillus delbrueckii',
    'Lactobacillus gasseri',
    'Lactobacillus jensenii',
    'Lactobacillus johnsonii',
    'Lactobacillus paracasei',
    'Lactobacillus plantarum',
    'Lactobacillus reuteri',
    'Lactobacillus rhamnosus',
    'Lactobacillus salivarius',
    'Listeria monocytogenes',
    'Micrococcus spp',
    'Neisseria gonorrhoeae',
    'Porphyromonas gingivalis',
    'Propionibacterium acnes',
    'Proteus',
    'Pseudomonas aeruginosa', 
    'Staphylococcus aureus', 
    'Staphylococcus epidermidis',
    'Staphylococcus saprophyticus',
    'Streptococcus agalactiae', 
    'Veillonella'
]

# ================== FUNCTIONS ==================

@st.cache_resource
def load_model():
    """
    Load model dengan prioritas:
    1. File lokal (jika ada) - untuk development
    2. Download dari Hugging Face - untuk production/cloud
    """
    model_path = None
    
    try:
        # Cek apakah file lokal ada
        if os.path.exists(LOCAL_MODEL_PATH):
            st.info(f"🏠 Using local model: `{LOCAL_MODEL_PATH}`")
            model_path = LOCAL_MODEL_PATH
        else:
            # Download dari Hugging Face jika file lokal tidak ada
            with st.spinner('📥 Downloading model from Hugging Face...'):
                model_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=MODEL_FILENAME,
                    cache_dir="./model_cache"
                )
                st.success(f"☁️ Model downloaded from Hugging Face")
        
        # Load model
        model_handler = ModelHandler(model_path, CLASS_NAMES)
        model_handler.load_model()
        return model_handler
        
    except FileNotFoundError:
        st.error(
            f"⚠️ **File model tidak ditemukan!**\n\n"
            f"**Lokasi lokal:** `{LOCAL_MODEL_PATH}`\n\n"
            f"**Hugging Face:** `{HF_REPO_ID}/{MODEL_FILENAME}`\n\n"
            f"**Solusi:**\n\n"
            f"1. Untuk development lokal: Pastikan file `{LOCAL_MODEL_PATH}` "
            f"ada di folder yang sama dengan `main.py`\n\n"
            f"2. Untuk production: Pastikan model sudah di-upload ke Hugging Face\n\n"
            f"   Cek di: https://huggingface.co/{HF_REPO_ID}"
        )
        return None
        
    except Exception as e:
        st.error(f"❌ **Error loading model:** {str(e)}")
        st.info(
            f"**Troubleshooting:**\n\n"
            f"1. Pastikan repo ID benar: `{HF_REPO_ID}`\n\n"
            f"2. Pastikan file model ada di repo: `{MODEL_FILENAME}`\n\n"
            f"3. Pastikan repository Hugging Face bersifat **public**\n\n"
            f"4. Cek versi TensorFlow kompatibel dengan model\n\n"
            f"Kunjungi: https://huggingface.co/{HF_REPO_ID}"
        )
        return None


def render_sidebar(model_handler):
    """Render sidebar dengan informasi aplikasi"""
    st.sidebar.title("🧬 Navigasi")
    st.sidebar.info(
        "Aplikasi ini mengklasifikasikan citra mikroskopis bakteri "
        "ke dalam **33 jenis bakteri** menggunakan Deep Learning "
        "dengan model Convolutional Neural Network (CNN)."
    )
    
    # Status model
    if model_handler:
        st.sidebar.success("✅ Model berhasil dimuat")
    else:
        st.sidebar.error("❌ Model tidak tersedia")
    
    # Menu selection
    menu = st.sidebar.radio(
        "Pilih Menu:",
        ["🔍 Prediksi Jenis Bakteri", "📊 Evaluasi Model"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Informasi")
    st.sidebar.info(
        f"**Total Kelas:** {len(CLASS_NAMES)} bakteri\n\n"
        f"**Model:** CNN-based classifier\n\n"
        f"**Input:** Citra mikroskopis RGB (224×224)\n\n"
        f"**Source:** Hugging Face / Local"
    )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption('© 2025 Bacteria Classification System')
    st.sidebar.caption('Developed with Streamlit & TensorFlow')
    st.sidebar.caption('Archie P.H.Sinaga')
    
    return menu


def render_prediction_page(model_handler):
    """Render halaman prediksi"""
    st.header("🔍 Prediksi Jenis Bakteri")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Citra")
        classifier = BacteriaClassificationApp(model_handler, CLASS_NAMES)
        image = classifier.display_classification()
    
    with col2:
        st.markdown("### 📋 Hasil Prediksi")
        
        if image:
            if st.button("🔍 Analisis Citra", type="primary", use_container_width=True):
                with st.spinner('🔄 Menganalisis citra mikroskopis...'):
                    try:
                        label, probability, exec_time = classifier.predict_bacteria(image)
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {str(e)}")
                        return
                
                # Validasi hasil prediksi
                if label is None:
                    st.error(
                        "⚠️ **Citra tidak valid!**\n\n"
                        "Citra yang diunggah kemungkinan bukan citra mikroskopis bakteri. "
                        "Harap unggah citra mikroskopis bakteri yang valid."
                    )
                    return
                
                # Confidence threshold check
                if probability < 0.7:
                    st.warning(
                        f"⚠️ **Confidence rendah!**\n\n"
                        f"Model tidak yakin dengan prediksi (confidence: **{probability*100:.2f}%**). "
                        "Coba citra lain atau periksa kualitas citra mikroskopis Anda."
                    )
                    st.info(f"Prediksi: **{label}**")
                else:
                    st.success(f"✅ **Prediksi Berhasil!**")
                    st.markdown(f"### 🦠 {label}")
                    st.metric("Confidence Score", f"{probability * 100:.2f}%")
                
                # Display bacteria description
                description = classifier.bacteria_descriptions.get(
                    label, 
                    "Deskripsi tidak tersedia untuk bakteri ini."
                )
                
                if probability < 0.7:
                    with st.expander("📖 Lihat Informasi Bakteri"):
                        st.info(description)
                else:
                    st.markdown("### 📖 Informasi Bakteri")
                    st.info(description)
                
                st.caption(f"⏱️ Waktu Eksekusi: {exec_time:.4f} detik")
        else:
            st.info("👆 Upload citra mikroskopis bakteri untuk memulai analisis")


def render_evaluation_page(model_handler):
    """Render halaman evaluasi model"""
    st.header("📊 Evaluasi Performa Model")
    
    evaluator = ModelEvaluation(model_handler, CLASS_NAMES)
    
    # Load test images
    try:
        test_images, test_labels = evaluator.get_test_images()
    except Exception as e:
        st.error(f"❌ Error saat memuat data test: {str(e)}")
        return
    
    # Validasi data test
    if test_images is None or len(test_images) == 0:
        st.warning("⚠️ Tidak ada gambar test yang ditemukan.")
        return
    
    st.info(f"📊 Total {len(test_images)} gambar test tersedia untuk evaluasi")
    st.markdown("---")
    
    # Tombol evaluasi
    if st.button("🚀 Jalankan Evaluasi", type="primary", use_container_width=True):
        with st.spinner('⏳ Sedang mengevaluasi model pada dataset test...'):
            try:
                exec_time = evaluator.evaluate_model(test_images, test_labels)
                st.success(f"✅ **Evaluasi selesai dalam {exec_time:.2f} detik**")
            except Exception as e:
                st.error(f"❌ Error saat evaluasi: {str(e)}")


# ================== MAIN APPLICATION ==================

def main():
    """Main application entry point"""
    
    # Header
    st.title("🔬 Klasifikasi Citra Mikroskopis Bakteri")
    st.markdown("---")
    
    # Load model
    model_handler = load_model()
    
    # Stop jika model gagal dimuat
    if model_handler is None:
        st.stop()
    
    # Render sidebar dan dapatkan menu selection
    menu = render_sidebar(model_handler)
    
    # Render halaman sesuai menu
    if menu == "🔍 Prediksi Jenis Bakteri":
        render_prediction_page(model_handler)
    elif menu == "📊 Evaluasi Model":
        render_evaluation_page(model_handler)


if __name__ == "__main__":
    main()