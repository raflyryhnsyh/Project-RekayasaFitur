import sys
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os
import cv2
from io import BytesIO

# Compatibility fix for loading Gradient Boosting model across scikit-learn versions
try:
    import sklearn._loss
    sys.modules['_loss'] = sklearn._loss
    for name in dir(sklearn._loss):
        if name.startswith('Half') or 'Error' in name or 'Loss' in name:
            setattr(sklearn._loss, f'Cy{name}', getattr(sklearn._loss, name))
except Exception:
    pass




# Try importing skimage hog
try:
    from skimage.feature import hog
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI vs Real Face Detector - Multi-Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .result-card {
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-top: 15px;
        margin-bottom: 25px;
        text-align: center;
    }
    
    .real-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 2px solid #10b981;
        color: #ecfdf5;
    }
    
    .ai-card {
        background: linear-gradient(135deg, #831843 0%, #9f1239 100%);
        border: 2px solid #f43f5e;
        color: #fff1f2;
    }
    
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .result-sub {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .model-badge {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "AdaBoost": "models/adaboost_model.pkl",
    "Gradient Boosting (GB)": "models/gradient_boosting_model.pkl",
    "K-Nearest Neighbors (KNN)": "models/knn_model.pkl",
    "Support Vector Machine (SVM)": "models/svm_model.pkl"
}

@st.cache_resource
def load_shared_transformers():
    """Load shared scaler and pca from models directory."""
    scaler_path = "models/scaler.pkl"
    pca_path = "models/pca.pkl"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
    return scaler, pca

@st.cache_resource
def load_pipeline_by_path(path_to_model):
    """Load model pipeline or estimator object using joblib."""
    if not os.path.exists(path_to_model):
        st.error(f"File model '{path_to_model}' tidak ditemukan!")
        return None
    try:
        data = joblib.load(path_to_model)
        if isinstance(data, dict):
            return data
        else:
            scaler, pca = load_shared_transformers()
            return {'scaler': scaler, 'pca': pca, 'model': data}
    except Exception as e:
        st.error(f"Gagal memuat model '{path_to_model}': {e}")
        return None

# ==============================================================================
# TAHAPAN PREPROCESSING SESUAI Preprocessing_Dataset_Final_Clean.ipynb & MODEL
# ==============================================================================

def step1_standardize(pil_img, target_size=(256, 256), quality=75):
    """
    Step 1 notebook: Standardize image dimensions to 256x256 RGB with JPEG quality 75.
    """
    img_rgb = pil_img.convert('RGB')
    img_resized = img_rgb.resize(target_size, Image.LANCZOS)
    
    buffer = BytesIO()
    img_resized.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    std_img = Image.open(buffer)
    size_kb = len(buffer.getvalue()) / 1024.0
    return std_img, size_kb

def step2_crop_and_normalize(pil_img, margin_ratio=0.2, target_size=(128, 128)):
    """
    Step 2 notebook: Convert to Grayscale, detect face, crop with 20% margin, resize to 128x128.
    """
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np
    
    face_found = False
    cropped_face = img_gray

    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face_found = True
                margin_x = int(margin_ratio * w)
                margin_y = int(margin_ratio * h)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(img_np.shape[1], x + w + margin_x)
                y2 = min(img_np.shape[0], y + h + margin_y)
                cropped_face = img_gray[y1:y2, x1:x2]
    except Exception:
        pass

    img_resized = cv2.resize(cropped_face, target_size)
    return img_resized, face_found

def step3_equalize_file_size(cropped_gray_np, target_kb=6.15, quality_range=(60, 95)):
    """
    Step 3 notebook: Compress cropped face image to target KB (~6.15 KB) via JPEG binary search.
    """
    pil_img = Image.fromarray(cropped_gray_np)
    low, high = quality_range
    best_buffer = None
    final_quality = 75
    
    for _ in range(8):
        mid = (low + high) // 2
        buffer = BytesIO()
        pil_img.save(buffer, 'JPEG', quality=mid)
        size_kb = buffer.tell() / 1024.0
        best_buffer = buffer
        final_quality = mid
        if abs(size_kb - target_kb) <= 0.5:
            break
        elif size_kb > target_kb:
            high = mid - 1
        else:
            low = mid + 1
            
    best_buffer.seek(0)
    equalized_img = Image.open(best_buffer)
    final_kb = len(best_buffer.getvalue()) / 1024.0
    return equalized_img, final_kb, final_quality

def step4_extract_features(equalized_img_pil, target_num_features=1926):
    """
    Step 4: Extract HOG (1764) + Grayscale Histogram (162) features.
    """
    img_gray_128 = np.array(equalized_img_pil.convert("L"))
    if img_gray_128.shape != (128, 128):
        img_gray_128 = cv2.resize(img_gray_128, (128, 128))
    
    if HAS_SKIMAGE:
        hog_feat = hog(
            img_gray_128, 
            orientations=9, 
            pixels_per_cell=(16, 16), 
            cells_per_block=(2, 2), 
            visualize=False
        )
    else:
        hog_cv = cv2.HOGDescriptor(_winSize=(128,128), _blockSize=(32,32), _blockStride=(16,16), _cellSize=(16,16), _nbins=9)
        hog_feat = hog_cv.compute(img_gray_128).flatten()
        
    hist_feat, _ = np.histogram(img_gray_128, bins=162, range=(0, 256))
    combined = np.hstack([hog_feat, hist_feat]).reshape(1, -1)
    
    if combined.shape[1] > target_num_features:
        combined = combined[:, :target_num_features]
    elif combined.shape[1] < target_num_features:
        combined = np.pad(combined, ((0, 0), (0, target_num_features - combined.shape[1])))
        
    return combined

def run_model_inference(pipeline_dict, features):
    """Executes Scaler -> PCA -> Model inference for a given model pipeline."""
    if pipeline_dict is None:
        return None, None
        
    scaler = pipeline_dict.get('scaler')
    pca = pipeline_dict.get('pca')
    model = pipeline_dict.get('model')
    
    X_proc = features
    if scaler is not None:
        X_proc = scaler.transform(X_proc)
    if pca is not None:
        X_proc = pca.transform(X_proc)
        
    prediction = model.predict(X_proc)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_proc)[0]
        except Exception:
            proba = None
    return prediction, proba

# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================

def main():
    st.title("🤖 AI vs 👤 Real Face Detector")
    st.markdown("Interface Prediksi Wajah AI vs Real dengan **Pilihan Model Machine Learning** (AdaBoost, Gradient Boosting, KNN, SVM).")
    st.markdown("---")
    
    # Sidebar Info & Model Selection
    with st.sidebar:
        st.header("⚙️ Pilih Model Machine Learning")
        
        selected_model_label = st.selectbox(
            "🤖 Model Classifier:",
            list(MODEL_FILES.keys()),
            index=0
        )
        
        target_model_file = MODEL_FILES[selected_model_label]
        pipeline_data = load_pipeline_by_path(target_model_file)
        
        if pipeline_data is not None:
            st.success(f"Model `{selected_model_label}` Aktif!")
            m_obj = pipeline_data.get('model')
            if m_obj:
                st.markdown(f"**Tipe Estimator:** `{type(m_obj).__name__}`")
        else:
            st.error(f"Gagal membaca `{target_model_file}`")
            
        st.markdown("---")
        compare_all = st.checkbox("📊 Bandingkan Semua Model Sekaligus", value=False)
        
        st.markdown("---")
        st.header("📋 Alur Preprocessing Notebook")
        st.markdown("""
        1. **Standarisasi**: Resize `256x256` RGB (JPEG Q75).
        2. **Crop & Margin 20%**: Deteksi wajah, crop dengan margin 20%, resize `128x128` Grayscale.
        3. **Equalize Size**: Kompresi ke target ukuran `~6.15 KB`.
        4. **Ekstraksi Fitur**: HOG `128x128` (1764) + Histogram Grayscale (162) = `1926` fitur.
        5. **Inferensi Model**: Scaler ➔ PCA ➔ Classifier (`0`=Real, `1`=AI).
        """)
        st.caption("TUBES RF - Multi-Model Classifier")

    if pipeline_data is None:
        return

    # Input Tabs
    tab_upload, tab_camera = st.tabs(["📁 Upload Gambar", "📷 Gunakan Kamera"])
    
    uploaded_file = None
    with tab_upload:
        uploaded_file = st.file_uploader("Unggah foto wajah (JPG, PNG, JPEG, WEBP)...", type=["jpg", "jpeg", "png", "webp"])
    with tab_camera:
        camera_file = st.camera_input("Ambil foto dari webcam")
        if camera_file is not None:
            uploaded_file = camera_file

    if uploaded_file is not None:
        col_main1, col_main2 = st.columns([1, 1], gap="medium")
        
        # Original Image
        raw_image = Image.open(uploaded_file)
        
        # Execute Preprocessing Steps
        with st.spinner("Menjalankan tahapan preprocessing dataset..."):
            # Step 1: Standardize
            std_img, std_kb = step1_standardize(raw_image, target_size=(256, 256), quality=75)
            
            # Step 2: Crop & Margin 20%
            cropped_gray_np, face_found = step2_crop_and_normalize(std_img, margin_ratio=0.2, target_size=(128, 128))
            
            # Step 3: Equalize File Size (~6.15 KB)
            equalized_img, final_kb, final_q = step3_equalize_file_size(cropped_gray_np, target_kb=6.15)
            
            # Step 4: Extract Features
            scaler_obj = pipeline_data.get('scaler')
            target_feats = getattr(scaler_obj, 'n_features_in_', 1926) if scaler_obj else 1926
            features = step4_extract_features(equalized_img, target_num_features=target_feats)
            
            # Step 5: Inference for selected model
            prediction, proba = run_model_inference(pipeline_data, features)
            
        with col_main1:
            st.subheader("🖼️ Preview Input Raw")
            st.image(raw_image, use_container_width=True, caption="Foto Asli Diunggah")
            
        with col_main2:
            st.subheader("📊 Hasil Deteksi AI vs Real")
            st.markdown(f'<div class="model-badge">Model Terpilih: {selected_model_label}</div>', unsafe_allow_html=True)
            
            # Interpret Result (0 = Real, 1 = AI)
            is_ai = False
            label_str = str(prediction).lower()
            if prediction == 1 or "ai" in label_str or "fake" in label_str:
                is_ai = True
            elif prediction == 0 or "real" in label_str:
                is_ai = False
            else:
                is_ai = bool(prediction)
                
            if is_ai:
                card_html = """
                <div class="result-card ai-card">
                    <div class="result-title">🤖 FOTO AI GENERATED</div>
                    <div class="result-sub">Sistem mendeteksi bahwa foto ini hasil buatan AI / Sintetis!</div>
                </div>
                """
            else:
                card_html = """
                <div class="result-card real-card">
                    <div class="result-title">👤 FOTO REAL / ASLI</div>
                    <div class="result-sub">Sistem mendeteksi bahwa foto ini adalah wajah manusia asli!</div>
                </div>
                """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Display Confidence / Probabilities (proba[0] = Real, proba[1] = AI)
            if proba is not None:
                st.markdown("##### 📈 Tingkat Keyakinan (Confidence Score):")
                c_real = float(proba[0]) * 100
                c_ai = float(proba[1]) * 100 if len(proba) > 1 else (100 - c_real)
                
                st.write(f"**Real / Human:** `{c_real:.2f}%`")
                st.progress(min(max(float(proba[0]), 0.0), 1.0))
                
                if len(proba) > 1:
                    st.write(f"**AI Generated:** `{c_ai:.2f}%`")
                    st.progress(min(max(float(proba[1]), 0.0), 1.0))

        # ======================================================================
        # PERBANDINGAN BANYAK MODEL SEKALIGUS (Jika Diaktifkan)
        # ======================================================================
        if compare_all:
            st.markdown("---")
            st.subheader("📊 Perbandingan Hasil Semua Model Sekaligus")
            st.caption("Membandingkan hasil deteksi dari AdaBoost, Gradient Boosting, KNN, dan SVM untuk foto yang sama:")
            
            comp_cols = st.columns(len(MODEL_FILES), gap="medium")
            
            for idx, (m_label, m_file) in enumerate(MODEL_FILES.items()):
                m_pipe = load_pipeline_by_path(m_file)
                m_pred, m_proba = run_model_inference(m_pipe, features)
                
                with comp_cols[idx]:
                    m_is_ai = (m_pred == 1)
                    res_tag = "🤖 AI GENERATED" if m_is_ai else "👤 REAL / ASLI"
                    border_col = "#f43f5e" if m_is_ai else "#10b981"
                    
                    st.markdown(f"""
                    <div style="background:#1e293b; border: 2px solid {border_col}; padding: 16px; border-radius: 12px; text-align: center;">
                        <h4 style="margin-bottom: 4px;">{m_label}</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; margin-bottom: 8px;">{res_tag}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if m_proba is not None:
                        real_p = float(m_proba[0]) * 100
                        ai_p = float(m_proba[1]) * 100 if len(m_proba) > 1 else (100 - real_p)
                        st.write(f"**Real:** `{real_p:.1f}%` | **AI:** `{ai_p:.1f}%`")
                        st.progress(min(max(float(m_proba[1 if m_is_ai else 0]), 0.0), 1.0))

        # ======================================================================
        # VISUALISASI TAHAPAN PREPROCESSING (Sesuai Notebook)
        # ======================================================================
        st.markdown("---")
        st.subheader("⚙️ Visualisasi Tahapan Preprocessing Notebook")
        st.caption("Proses bertahap transformasi foto sesuai logika `Preprocessing_Dataset_Final_Clean.ipynb`:")
        
        p_col1, p_col2, p_col3, p_col4 = st.columns(4, gap="medium")
        
        with p_col1:
            st.markdown("#### 1. Standarisasi")
            st.image(std_img, use_container_width=True, caption="256x256 RGB (JPEG Q75)")
            st.info(f"**Ukuran File:** {std_kb:.1f} KB\n**Dimensi:** 256x256")
            
        with p_col2:
            st.markdown("#### 2. Crop & Margin 20%")
            st.image(cropped_gray_np, use_container_width=True, caption="128x128 Grayscale (Cropped)")
            st.info(f"**Deteksi Wajah:** {'Berhasil ✅' if face_found else 'Fallback Full ℹ️'}\n**Margin Ratio:** 20%")
            
        with p_col3:
            st.markdown("#### 3. Equalize Size")
            st.image(equalized_img, use_container_width=True, caption=f"JPEG Q{final_q} Equalized")
            st.info(f"**Target Size:** ~6.15 KB\n**Ukuran Hasil:** {final_kb:.2f} KB")
            
        with p_col4:
            st.markdown("#### 4. Fitur & Inferensi")
            st.markdown(f"""
            - **HOG + Hist (Exact):** `{features.shape[1]}` fitur
            - **Model:** `{selected_model_label}`
            - **SVM/KNN/GB Class:** `{prediction}`
            """)
            st.success("Inferensi Siap ✅")

if __name__ == "__main__":
    main()
