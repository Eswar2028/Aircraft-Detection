import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gc

# Increase PIL image size limit to prevent DecompressionBombError on large images
Image.MAX_IMAGE_PIXELS = None

# Set page config for a wider layout
st.set_page_config(page_title="Aircraft Detection Web App", layout="wide")

st.title("🛫 Aircraft Detection with YOLOv8 🛬")
st.write("Upload an image of an aircraft, and the YOLOv8 model will detect and draw bounding boxes around it.")

def resize_image(image, max_size=1280):
    """Resizes the image if it exceeds max_size to save memory."""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.LANCZOS)
    return image

# --- Model Loading ---
@st.cache_resource()
def load_model():
    """Loads the YOLOv8 model. Cached to avoid loading on every run."""
    try:
        model = YOLO("Aircarft-Detection-YOLO.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner("Loading YOLOv8 Model..."):
    model = load_model()

if model is None:
    st.stop()


# --- File Upload ---
uploaded_file = st.file_uploader("📥 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and resize image to prevent OOM
        image = Image.open(uploaded_file).convert('RGB')
        image = resize_image(image)
        
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
             st.subheader("Original Image")
             st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button("Detect Aircraft 🔍"):
            with st.spinner("Running Inference..."):
                # Run YOLOv8 inference
                # Model predicts on the PIL Image or numpy array
                results = model(image)
                
                # Get the result image with boxes drawn
                # Results is a list, we take the first item since we only passed one image
                res = results[0]
                res_plotted = res.plot() # BGR numpy array
                
                # Convert BGR to RGB for correct displaying in Streamlit via PIL
                res_rgb = res_plotted[:, :, ::-1]
                detected_image = Image.fromarray(res_rgb)
                
                with col2:
                    st.subheader("Detection Result")
                    st.image(detected_image, caption='Detected Aircrafts', use_container_width=True)
                
                # Expandable Details
                with st.expander("Show Detection Details 📊"):
                     # Parse results
                     boxes = res.boxes
                     if len(boxes) > 0:
                         st.write(f"**Total Detections:** {len(boxes)}")
                         for i, box in enumerate(boxes):
                             class_id = int(box.cls[0].item())
                             class_name = model.names[class_id]
                             confidence = float(box.conf[0].item())
                             st.write(f"- Detection {i+1}: Class: **{class_name}**, Confidence: **{confidence:.2f}**")
                     else:
                         st.write("No aircraft detected in this image.")
                
                # Cleanup to free memory
                del results
                del res
                del res_plotted
                del res_rgb
                del detected_image
                gc.collect()
                         
    except Exception as e:
        st.error(f"Error processing the image: {e}")
