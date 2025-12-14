from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from pyarrow import null
import streamlit as st
from PIL import Image

# NOTE: ultralytics may expose YOLO differently selon la version.
# This code expects `from ultralytics import YOLO` available.
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - runtime environment dependent
    YOLO = None  # type: ignore

import streamlit_webrtc as webrtc
import av

if st.query_params.get("mobile", [""])[0] == "":
    st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1">', unsafe_allow_html=True)


# ---------------------------
# Configuration globale
# ---------------------------
st.set_page_config(page_title="YOLOv8 AI", layout="wide")
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("app")

# Constants
DEFAULT_MODEL_PATH = "runs/detect/train2/weights/best.pt"
FALLBACK_MODEL_NAME = "yolov8n.pt"
TEMP_DIR = tempfile.gettempdir()
VIDEO_OUTPUT_DIR = os.path.join(TEMP_DIR, "yolo_outputs")
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


# ---------------------------
# Utilities / Helpers
# ---------------------------
def seconds_elapsed(start: float) -> float:
    """Return seconds elapsed since start, with safety."""
    elapsed = time.time() - start
    return max(elapsed, 1e-6)


def safe_remove(path: str) -> None:
    """Remove a file if it exists; ignore errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
            LOG.debug("Removed file: %s", path)
    except Exception:
        LOG.exception("Failed to remove temporary file: %s", path)


# ---------------------------
# Model loading & management
# ---------------------------
@st.cache_resource
def load_model(model_path: Optional[str] = None, device: Optional[str] = None):
    """
    Load a YOLO model with caching.
    - model_path: path to weights file or None to use fallback.
    - device: 'cpu' or 'cuda' or None (auto).
    Returns: model instance or None (and logs error).
    """
    if YOLO is None:
        LOG.error("Ultralytics YOLO not installed or import failed.")
        return None

    model_path = model_path or DEFAULT_MODEL_PATH

    # Prefer provided path if exists else fallback to official
    if os.path.exists(model_path):
        chosen = model_path
    else:
        chosen = FALLBACK_MODEL_NAME

    try:
        LOG.info("Loading model: %s (device=%s)", chosen, device)
        model = YOLO(chosen)  # type: ignore
        if device is not None:
            try:
                model.to(device)  # type: ignore
            except Exception:
                LOG.debug("model.to(%s) failed or unsupported", device)
        return model
    except Exception:
        LOG.exception("Failed to load model %s", chosen)
        return None


def model_predict(model, image_rgb: np.ndarray, conf: float, max_det: int):
    """
    Safe wrapper to call model on an RGB numpy image.
    Returns results object or None on failure.
    """
    if model is None:
        LOG.error("No model available for prediction.")
        return None
    try:
        # ultralytics models accept numpy arrays in RGB
        results = model(image_rgb, conf=conf, max_det=max_det)  # type: ignore
        return results
    except TypeError:
        # older/newer APIs might require predict()
        try:
            results = model.predict(image_rgb, conf=conf, max_det=max_det)  # type: ignore
            return results
        except Exception:
            LOG.exception("Model prediction failed with both call and predict().")
            return None
    except Exception:
        LOG.exception("Model prediction failed.")
        return None


# ---------------------------
# Image / Frame processing
# ---------------------------
def pil_to_rgb_array(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB numpy array."""
    return np.asarray(pil_img.convert("RGB"))


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB numpy array (uint8)."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR for OpenCV write/display if needed."""
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def annotate_results_to_bgr(results) -> Optional[np.ndarray]:
    """
    Given ultralytics results, return an OpenCV BGR image ready to write/display.
    The results[0].plot() is typically RGB numpy array.
    """
    try:
        vis = results[0].plot()
        if isinstance(vis, np.ndarray):
            # assume vis is RGB
            return rgb_to_bgr(vis)
        # If PIL returned, convert
        if hasattr(vis, "convert"):
            vis = np.asarray(vis.convert("RGB"))
            return rgb_to_bgr(vis)
    except Exception:
        LOG.exception("Failed to create annotated image from results.")
    return None


# ---------------------------
# Video processing
# ---------------------------
def safe_int(value, fallback: int) -> int:
    """Convert numeric to int with fallback for zero/None."""
    try:
        ival = int(value)
        return ival if ival > 0 else fallback
    except Exception:
        return fallback


def process_video_file(
    model,
    input_path: str,
    output_path: Optional[str],
    conf: float,
    max_det: int,
    preview_callback=None,
) -> Tuple[int, Optional[str]]:
    """
    Process a video file with YOLO model.
    - input_path: path to input video
    - output_path: path to write annotated video (if None, create temp file)
    - preview_callback: optional callback(frame_bgr, frame_index) to preview frames in UI
    Returns (frame_count, output_path_or_none)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        LOG.error("Cannot open video: %s", input_path)
        return 0, None

    fps = safe_int(cap.get(cv2.CAP_PROP_FPS), 30)
    width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 640)
    height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 480)

    if output_path is None:
        fd, out_name = tempfile.mkstemp(suffix=".mp4", dir=VIDEO_OUTPUT_DIR)
        os.close(fd)
        output_path = out_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # convert to RGB for model
            frame_rgb = bgr_to_rgb(frame_bgr)
            results = model_predict(model, frame_rgb, conf=conf, max_det=max_det)
            annotated_bgr = None
            if results is not None:
                annotated_bgr = annotate_results_to_bgr(results)

            # If annotation failed, fallback to original frame
            if annotated_bgr is None:
                annotated_bgr = frame_bgr

            out.write(annotated_bgr)
            frame_count += 1

            if preview_callback is not None:
                # pass a copy to avoid mutation issues
                try:
                    preview_callback(annotated_bgr.copy(), frame_count)
                except Exception:
                    LOG.exception("Preview callback failed.")
    except Exception:
        LOG.exception("Error during video processing.")
    finally:
        cap.release()
        out.release()

    LOG.info("Processed %d frames into %s", frame_count, output_path)
    return frame_count, output_path


# ---------------------------
# Streamlit UI components
# ---------------------------
def show_header() -> None:
    """Render header and CSS (minimal CSS kept)."""
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .stApp { 
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff !important; 
        }
        h1 { 
            font-size: 2.2rem !important; 
            font-weight: 700 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
            text-align: center; 
            margin-bottom: 0.5rem; 
        }

        .stButton > button {
            background-color: #1E88E5 !important;
            color: white !important;
            border: 1px solid #1E88E5 !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
        }
        
        .stButton > button:hover {
            background-color: #1565C0 !important;
            border-color: #1565C0 !important;
            color: white !important;
        }
        
        .card { 
            background: rgba(255,255,254,0.04); 
            border-radius: 12px; 
            padding: 1rem; 
            margin: 0.7rem 0;
            color: white; 
        } 
        
        /* Style pour les labels des file uploaders */
        section[data-testid="stFileUploader"] > label > div > p,
        div[data-testid="stFileUploader"] > label > div > p {
            color: #1E88E5 !important;  
            font-weight: 500 !important;
            font-size: 16px !important;
        }
        
        /* Style pour le bouton "Browse files" */
        section[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploader"] button {
            background-color: #1E88E5 !important;
            color: white !important;
            border: 1px solid #1E88E5 !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            margin-top: 8px !important;
        }
        
        section[data-testid="stFileUploader"] button:hover,
        div[data-testid="stFileUploader"] button:hover {
            background-color: #1565C0 !important;
            border-color: #1565C0 !important;
            color: white !important;
        }
        
        section[data-testid="stFileUploader"] button span,
        div[data-testid="stFileUploader"] button span {
            color: white !important;
            font-weight: 500 !important;
        }

        /* Style pour le bouton "Take Photo" */
        div[data-testid="stCameraInput"] button {
            background-color: #1E88E5 !important;
            color: white !important;
            border: 1px solid #1E88E5 !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            margin: 10px auto !important;
            display: block !important;
        }
        
        div[data-testid="stCameraInput"] button:hover {
            background-color: #1565C0 !important;
            border-color: #1565C0 !important;
            color: white !important;
        }
        
        div[data-testid="stCameraInput"] button span {
            color: white !important;
            font-weight: 500 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 8px 8px 0px 0px;
            gap: 5px;
            padding: 10px 16px;
            border: 1px solid #E0E0E0;
            border-bottom: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1E88E5 !important;
            color: white !important;
            border-color: #1E88E5 !important;
        }
        
        .stTabs [data-baseweb="tab"] > div > p {
            color: #1E88E5 !important;  /* Bleu pour les onglets inactifs */
            font-weight: 500 !important;
            font-size: 16px !important;
        }
        
        .stTabs [aria-selected="true"] > div > p {
            color: white !important;  /* Blanc pour l'onglet actif */
            font-weight: 600 !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
            background-color: rgba(30, 136, 229, 0.1) !important;
            border-color: rgba(30, 136, 229, 0.3) !important;
        }

        /* Style pour le bouton "Télécharger la vidéo annotée" */
        div[data-testid="stDownloadButton"] button {
            background-color: #1E88E5 !important;
            color: white !important;
            border: 1px solid #1E88E5 !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            margin: 10px auto !important;
            display: block !important;
        }

        div[data-testid="stDownloadButton"] button:hover {
            background-color: #1565C0 !important;
            border-color: #1565C0 !important;
            color: white !important;
        }

        div[data-testid="stDownloadButton"] button span {
            color: white !important;
            font-weight: 500 !important;
        }

        /* Label de la caméra (st.camera_input) même style que file_uploader */
        div[data-testid="stCameraInput"] > label > div > p {
            color: #1E88E5 !important;
            font-weight: 500 !important;
            font-size: 16px !important;
        }

        div.stTabs [data-baseweb="tab"] {
            flex: 1 1 auto;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        @media (max-width: 768px) {
            div.stTabs [data-baseweb="tab"] {
                font-size: 0.7rem;
                padding: 0.5rem 0.3rem;
            }
 
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style='text-align:center; padding: 1rem 0;'>
        <h1>YOLOv8 AI — Real-time Object Detection and Classification — Project</h1>
        <p style='color:#b0b0d4;margin-top:-8px'>Projet de Synthèse — Data Science</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> Tuple[float, int, Optional[str]]:
    """Sidebar widgets for model controls and device selection."""
    st.sidebar.markdown("### ⚙️ Paramètres IA")
    conf_slider = st.sidebar.slider("🎯 Confiance (%)", 1, 99, 25)
    conf = conf_slider / 100.0
    max_det = st.sidebar.slider("📊 Max objets par image", 1, 100, 50)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Modèle and Device")
    model_path_input = st.sidebar.text_input(
        "Chemin du modèle", DEFAULT_MODEL_PATH
    )
    device = st.sidebar.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
    device_arg = None if device == "auto" else device

    return conf, max_det, device_arg if model_path_input == "" else model_path_input



def tab_webcam(model, conf: float, max_det: int) -> None:
    """
    Tab for webcam / camera input.
    We prefer st.camera_input() because it's browser friendly.
    """
    st.markdown(
        '<div class="card"><h3>🎥 Image par Caméra</h3></div>', 
        unsafe_allow_html=True
    )
    
    # SIMPLE : Juste st.camera_input() avec VOTRE texte
    cam_file = st.camera_input(
        "Autorise l'accès à la camera et prends une photo"
    )
    
    if cam_file is not None:
        try:
            pil_img = Image.open(cam_file)
            rgb = pil_to_rgb_array(pil_img)
            
            with st.spinner("Analyse..."):
                results = model_predict(model, rgb, conf=conf, max_det=max_det)
                annotated_bgr = annotate_results_to_bgr(results) if results is not None else None
                
                # VOTRE VISUALISATION EXACTE
                if annotated_bgr is None:
                    st.image(rgb, caption="Image (originale)", width="stretch")
                else:
                    # annotated_bgr -> convert back to RGB for st.image
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Analyse IA", width="stretch")
                    
        except Exception:
            LOG.exception("Failed to process camera_input image.")
            st.error("Erreur lors du traitement de l'image prise par la caméra.")


def tab_images(model, conf: float, max_det: int) -> None:
    st.markdown(
        '<div class="card"><h3>🖼️ Traitement Image</h3></div>', 
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("📁 Glisse une image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])

    if not uploaded:
        return

    try:
        with st.spinner("Analyse de l'image..."):
            pil_img = Image.open(uploaded)
            rgb = pil_to_rgb_array(pil_img)
            results = model_predict(model, rgb, conf=conf, max_det=max_det)

            if results is None:
                st.error("L'analyse a échoué. Vérifiez le modèle et les entrées.")
                return

            annotated_bgr = annotate_results_to_bgr(results)
            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb, caption="Source", width="stretch")
            with col2:
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB) if annotated_bgr is not None else rgb
                st.image(annotated_rgb, caption="Analyse IA", width="stretch")
    except Exception:
        LOG.exception("Erreur lors du traitement de l'image uploadée.")
        st.error("Impossible de traiter l'image uploadée. Assurez-vous que le fichier est une image valide.")


def tab_videos(model, conf: float, max_det: int) -> None:
    st.markdown(
        '<div class="card"><h3>🎥 Traitement Vidéo</h3></div>', 
        unsafe_allow_html=True
    )
    video_file = st.file_uploader("📹 Vidéo MP4 (max ~100MB recommandé,)", type=["mp4"])

    if not video_file:
        return

    temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_input_fd)
    try:
        with open(temp_input_path, "wb") as f:
            f.write(video_file.read())

        st.info("Fichier vidéo reçu. Préparation du traitement...")
        preview = st.empty()
        progress = st.progress(0)
        status_text = st.empty()

        def preview_cb(frame_bgr: np.ndarray, frame_index: int):
            # Update preview and progress every N frames to reduce UI thrash
            if frame_index % 10 == 0:
                preview.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                # progress approximate (not accurate without frame count)
                progress.progress(min(100, int((frame_index % 100))))

        # Construct a safe output path
        out_fname = f"annotated_{int(time.time())}.mp4"
        out_path = os.path.join(VIDEO_OUTPUT_DIR, out_fname)

        start_time = time.time()
        frame_count, generated_path = process_video_file(
            model,
            temp_input_path,
            out_path,
            conf=conf,
            max_det=max_det,
            preview_callback=preview_cb,
        )
        duration = seconds_elapsed(start_time)

        if generated_path:
            status_text.success(f"Terminé — {frame_count} frames traitées en {duration:.1f}s")
            st.video(generated_path)
            st.markdown(f"**Fichier de sortie**: `{generated_path}`")
            # Offer download
            with open(generated_path, "rb") as fh:
                st.download_button(
                    label="Télécharger la vidéo annotée",
                    data=fh,
                    file_name=os.path.basename(generated_path),
                    mime="video/mp4",
                )
        else:
            st.error("Le traitement vidéo a échoué.")
    except Exception:
        LOG.exception("Erreur lors du traitement du fichier vidéo uploadé.")
        st.error("Impossible de traiter la vidéo uploadée.")
    finally:
        safe_remove(temp_input_path)


def tab_analytics() -> None:
    st.markdown(
        '<div class="card"><h3>📊 Dashboard Performances</h3></div>', 
        unsafe_allow_html=True
    )
    import pandas as pd 

    df_top = pd.DataFrame(
        {
            "Classes": [
                "Avion(aeroplane)", "Bicyclette(bicycle)", "Oiseau(bird)", "Bâteau(boat)", "Bouteille(boottle)", "Bus(bus)", "Voiture(car)", 
                "Chat(cat)", "Chaise(chair)", "Vache(cow)", "Table à manger(diningtable)", "Chien(dog)", "Cheval(horse)", "Moto(motorbike)", 
                "Personne(person)", "Plante en pot(pottedplant)", "Mouton(sheep)", "Canapé(sofa)", "Train(train)", "Téléviseur(tvmonitor)"
            ],
            "mAP50": [
                "82.3%", "72.8%", "66.6%", "53.5%", "35.3%", "81.0%","83.0%", 
                "80.9%", "55.1%", "61.8%", "73.6%","53.2%", "77.8%", "73.3%", 
                "70.4%", "44.9%",  "74.3%", "70.7%", "92.8%", "84.7%"
            ]
        }
    )

    st.subheader("🏆 Toutes Classes")
    st.dataframe(df_top, width="stretch")


def tab_realtime(model, conf: float, max_det: int) -> None:
    st.markdown(
        '<div class="card"><h3>🔴 Détection Temps Réel</h3><p>Webcam live automatique</p></div>', 
        unsafe_allow_html=True    
    )

    class VideoProcessor:
        def __init__(self): 
            self.conf = conf
            self.max_det = max_det
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Conversion + YOLO AVEC LES PARAMS
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model_predict(model, frame_rgb, self.conf, self.max_det) 
            annotated = annotate_results_to_bgr(results) if results else img
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    
    # WEBCAM LIVE avec PARAMS transmis
    ctx = webrtc.webrtc_streamer(
        key="realtime-key",
        video_processor_factory= VideoProcessor,  
        media_stream_constraints={
            'video': {'width': {'ideal': 640}, 'height': {'ideal': 480}}
        }
    )

    # MISE À JOUR PARAMS EN TEMPS RÉEL
    if ctx and ctx.state.playing and ctx.video_processor:
        ctx.video_processor.conf = conf
        ctx.video_processor.max_det = max_det
    
# ---------------------------
# Main app
# ---------------------------
def main() -> None:
    show_header()

    # Sidebar controls (returns conf, max_det, model_path_or_device)
    conf, max_det, model_path_or_device = sidebar_controls()

    model = None
    device_arg = None
    model_path = None
    # Determine if user provided a custom model path (string) or a device selection
    if isinstance(model_path_or_device, str) and os.path.exists(model_path_or_device):
        model_path = model_path_or_device
    elif isinstance(model_path_or_device, str) and model_path_or_device != DEFAULT_MODEL_PATH:
        # The user entered a path that doesn't exist; we'll still pass it to loader and let it fail gracefully.
        model_path = model_path_or_device
    else:
        # model_path_or_device may be a device (cpu/cuda) or None
        device_arg = model_path_or_device if model_path_or_device in ("cpu", "cuda") else None

    with st.spinner("Chargement du modèle..."):
        model = load_model(model_path or DEFAULT_MODEL_PATH, device=device_arg)
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez les logs côté serveur.")
        LOG.error("Model unavailable after load attempt.")
        
    st.success("Modèle prêt (chargement terminé)", icon="✅")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔴 Détection Live", "📸 Caméra", "📁 Images", "🎥 Vidéos", "📈 Analytics"])

    with tab1:
        tab_realtime(model, conf, max_det)

    with tab2:
        tab_webcam(model, conf, max_det)

    with tab3:
        tab_images(model, conf, max_det)

    with tab4:
        tab_videos(model, conf, max_det)

    with tab5:
        tab_analytics()
        

    # Footer
    st.markdown(
    """
    <div style="margin-top:1.2rem;padding:1rem;border-radius:10px;background:rgba(0,0,0,0.5);color:white;text-align:center">
        <strong>Projet de Synthèse</strong> • Maliki Bouba & Ngounou Florian • Touza Isaac
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
