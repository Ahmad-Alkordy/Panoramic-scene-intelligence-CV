import streamlit as st
import cv2
import numpy as np
from skimage import segmentation, color
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="Smart Panorama System", layout="wide", page_icon="🌐")

# Load model once and cache it
@st.cache_resource
def load_classifier():
    try:
        return tf.keras.models.load_model('panorama_efficientnet_final.keras')
    except Exception as e:
        return None

model = load_classifier()
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Helper to load uploaded images
def load_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌐 Smart Panorama")
st.sidebar.markdown("Navigate the Computer Vision Pipeline:")
app_mode = st.sidebar.radio("Select Step:", 
    ["1. Image Preprocessing", 
     "2. Panorama Stitching", 
     "3. Image Segmentation", 
     "4. Scene Classification"]
)

# ==========================================
# 1. IMAGE PREPROCESSING
# ==========================================
if app_mode == "1. Image Preprocessing":
    st.title("✨ Image Preprocessing & Filtering")
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        _, img_rgb = load_uploaded_image(uploaded_file)
        
        st.subheader("Filter Tuning")
        col1, col2 = st.columns(2)
        with col1:
            ksize = st.slider("Kernel Size (Gaussian & Median)", 3, 15, 5, step=2)
        with col2:
            sigma = st.slider("Sigma (Gaussian)", 0.5, 5.0, 1.5)
            
        gauss = cv2.GaussianBlur(img_rgb, (ksize, ksize), sigmaX=sigma)
        median = cv2.medianBlur(img_rgb, ksize)
        
        c1, c2, c3 = st.columns(3)
        c1.image(img_rgb, caption="Original Image", use_container_width=True)
        c2.image(gauss, caption=f"Gaussian Filter ({ksize}x{ksize})", use_container_width=True)
        c3.image(median, caption=f"Median Filter ({ksize}x{ksize})", use_container_width=True)

# ==========================================
# 2. PANORAMA STITCHING
# ==========================================
elif app_mode == "2. Panorama Stitching":
    st.title("🗺️ SIFT Panorama Stitching")
    uploaded_files = st.file_uploader("Upload overlapping images (at least 2)", type=['jpg', 'png'], accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) >= 2:
        imgs_bgr = []
        cols = st.columns(len(uploaded_files))
        for i, f in enumerate(uploaded_files):
            bgr, rgb = load_uploaded_image(f)
            # Resize for performance just like the notebook
            bgr = cv2.resize(cv2.GaussianBlur(bgr, (5,5), 1.5), (400, 300))
            imgs_bgr.append(bgr)
            cols[i].image(rgb, caption=f"Image {i+1}", use_container_width=True)
            
        if st.button("Stitch Images", type="primary"):
            with st.spinner("Stitching using OpenCV Stitcher..."):
                stitcher = cv2.Stitcher_create()
                status, pano = stitcher.stitch(imgs_bgr)
                
                if status == cv2.Stitcher_OK:
                    st.success("Panorama stitched successfully!")
                    st.image(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.error("Not enough matching keypoints found to stitch these images.")

# ==========================================
# 3. IMAGE SEGMENTATION
# ==========================================
elif app_mode == "3. Image Segmentation":
    st.title("🎨 Image Segmentation (SLIC)")
    uploaded_file = st.file_uploader("Upload an image to segment...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        _, img_rgb = load_uploaded_image(uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        n_segments = col1.slider("Number of Segments", 10, 200, 80)
        compactness = col2.slider("Compactness", 1.0, 50.0, 10.0)
        sigma = col3.slider("Sigma (Smoothing)", 0.0, 5.0, 1.0)
        
        if st.button("Run Segmentation"):
            with st.spinner("Generating Superpixels..."):
                seg_slic = segmentation.slic(img_rgb, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)
                slic_bound = segmentation.mark_boundaries(img_rgb, seg_slic, color=(1, 0.3, 0))
                
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original Image", use_container_width=True)
                c2.image(slic_bound, caption=f"SLIC ({seg_slic.max()} regions)", use_container_width=True)

# ==========================================
# 4. SCENE CLASSIFICATION
# ==========================================
elif app_mode == "4. Scene Classification":
    st.title("🎯 Scene Classification (EfficientNetB0)")
    
    if model is None:
        st.warning("⚠️ Could not load `panorama_efficientnet_final.keras`. Make sure the file is in the same directory as this script.")
    else:
        uploaded_file = st.file_uploader("Upload an image to classify...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            img_bgr, img_rgb = load_uploaded_image(uploaded_file)
            
            c1, c2 = st.columns([1, 1])
            c1.image(img_rgb, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Classifying..."):
                # Preprocess for EfficientNet
                img_resized = cv2.resize(img_bgr, (224, 224))
                img_proc = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                img_proc = preprocess_input(img_proc)
                
                # Predict
                probs = model.predict(np.expand_dims(img_proc, 0), verbose=0)[0]
                top3_idx = np.argsort(probs)[::-1][:3]
                
                c2.subheader("Prediction Results")
                best_pred = CLASSES[top3_idx[0]]
                c2.success(f"**{best_pred.capitalize()}** ({probs[top3_idx[0]]*100:.1f}%)")
                
                # Display Top 3
                for i in top3_idx:
                    c2.progress(float(probs[i]), text=f"{CLASSES[i].capitalize()}: {probs[i]*100:.1f}%")