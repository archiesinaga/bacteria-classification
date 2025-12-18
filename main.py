import streamlit as st
from models.loader import ModelHandler
from prediksi import BacteriaClassificationApp
from evaluasi import ModelEvaluation

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Bakteri",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ UBAH: Path model sesuai dengan nama file Anda
model_path = 'best_model.keras'

# Daftar 33 kelas bakteri - SESUAIKAN urutan dengan training model Anda
class_names = [
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

# Load model dengan caching
@st.cache_resource
def load_model():
    """Load model dengan caching untuk performa lebih baik."""
    model_handler = ModelHandler(model_path, class_names)
    model_handler.load_model()
    return model_handler

# Main application
if __name__ == "__main__":
    # Header
    st.title("🔬 Klasifikasi Citra Mikroskopis Bakteri")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🧬 Navigasi")
    st.sidebar.info(
        "Aplikasi ini mengklasifikasikan citra mikroskopis bakteri "
        "ke dalam **33 jenis bakteri** menggunakan Deep Learning "
        "dengan model Convolutional Neural Network (CNN)."
    )
    
    # Load model
    try:
        model_handler = load_model()
        st.sidebar.success("✅ Model berhasil dimuat")
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ File model tidak ditemukan")
        st.error(
            f"⚠️ **File model tidak ditemukan!**\n\n"
            f"Pastikan file `{model_path}` ada di folder root project.\n\n"
            f"**Lokasi yang dicari:** `{model_path}`\n\n"
            f"**Solusi:**\n"
            f"1. Pastikan file `{model_path}` ada di folder yang sama dengan `main.py`\n"
            f"2. Atau ubah `model_path` di `main.py` sesuai lokasi file model Anda"
        )
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat model")
        st.error(
            f"⚠️ **Error saat memuat model:**\n\n"
            f"```\n{str(e)}\n```\n\n"
            f"**Kemungkinan penyebab:**\n"
            f"1. File model corrupt atau tidak kompatibel\n"
            f"2. Versi TensorFlow tidak sesuai\n"
            f"3. Format file bukan `.keras` yang valid"
        )
        st.stop()
    
    # Menu selection
    menu = st.sidebar.radio(
        "Pilih Menu:",
        ["🔍 Prediksi Jenis Bakteri", "📊 Evaluasi Model"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Informasi")
    st.sidebar.info(
        f"**Total Kelas:** {len(class_names)} bakteri\n\n"
        f"**Model:** CNN-based classifier\n\n"
        f"**Input:** Citra mikroskopis RGB (224×224)\n\n"
        f"**File Model:** `{model_path}`"
    )
    
    # ========== MENU PREDIKSI ==========
    if menu == "🔍 Prediksi Jenis Bakteri":
        st.header("🔍 Prediksi Jenis Bakteri")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Citra")
            classifier = BacteriaClassificationApp(model_handler, class_names)
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
                            st.stop()
                    
                    if label is None:
                        st.error(
                            "⚠️ **Citra tidak valid!**\n\n"
                            "Citra yang diunggah kemungkinan bukan citra mikroskopis bakteri. "
                            "Harap unggah citra mikroskopis bakteri yang valid."
                        )
                    elif probability < 0.7:
                        st.warning(
                            f"⚠️ **Confidence rendah!**\n\n"
                            f"Model tidak yakin dengan prediksi (confidence: **{probability*100:.2f}%**). "
                            "Coba citra lain atau periksa kualitas citra mikroskopis Anda."
                        )
                        st.info(f"Prediksi: **{label}**")
                        
                        # Tetap tampilkan deskripsi meskipun confidence rendah
                        description = classifier.bacteria_descriptions.get(
                            label, 
                            "Deskripsi tidak tersedia untuk bakteri ini."
                        )
                        with st.expander("📖 Lihat Informasi Bakteri"):
                            st.info(description)
                        
                        st.caption(f"⏱️ Waktu Eksekusi: {exec_time:.4f} detik")
                    else:
                        st.success(f"✅ **Prediksi Berhasil!**")
                        
                        # Display prediction
                        st.markdown(f"### 🦠 {label}")
                        st.metric("Confidence Score", f"{probability * 100:.2f}%")
                        
                        # Display description
                        description = classifier.bacteria_descriptions.get(
                            label, 
                            "Deskripsi tidak tersedia untuk bakteri ini."
                        )
                        
                        st.markdown("### 📖 Informasi Bakteri")
                        st.info(description)
                        
                        st.caption(f"⏱️ Waktu Eksekusi: {exec_time:.4f} detik")
            else:
                st.info("👆 Upload citra mikroskopis bakteri untuk memulai analisis")
    
    # ========== MENU EVALUASI (SIMPLIFIED) ==========
    elif menu == "📊 Evaluasi Model":
        st.header("📊 Evaluasi Performa Model")
        
        evaluator = ModelEvaluation(model_handler, class_names)
        
        # Load test images tanpa menampilkan detail berlebihan
        try:
            test_images, test_labels = evaluator.get_test_images()
        except Exception as e:
            st.error(f"❌ Error saat memuat data test: {str(e)}")
            test_images, test_labels = None, None
        
        # Jika gambar berhasil dimuat, tampilkan tombol evaluasi
        if test_images is not None and len(test_images) > 0:
            st.markdown("---")
            
            if st.button("🚀 Jalankan Evaluasi", type="primary", use_container_width=True):
                with st.spinner('⏳ Sedang mengevaluasi model pada dataset test...'):
                    try:
                        exec_time = evaluator.evaluate_model(test_images, test_labels)
                        st.success(f"✅ **Evaluasi selesai dalam {exec_time:.2f} detik**")
                    except Exception as e:
                        st.error(f"❌ Error saat evaluasi: {str(e)}")
        elif test_images is not None and len(test_images) == 0:
            st.warning("⚠️ Tidak ada gambar test yang ditemukan.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption('© 2025 Bacteria Classification System')
    st.sidebar.caption('Developed with Streamlit & TensorFlow')
    st.sidebar.caption('Archie P.H.Sinaga')