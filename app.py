import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from mtcnn import MTCNN
import tempfile
from PIL import Image

# Constants (from model training notebook)
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Loads model and feature extractor with caching
@st.cache_resource
def load_model():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    feature_extractor = keras.Model(inputs, outputs, name="feature_extractor")

    class_vocab = ['fake', 'real']
    lstm_model = keras.models.load_model("deepfake_detector_model.keras")
    
    return feature_extractor, lstm_model, class_vocab

# Video processing functions
def get_center(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x//2)-(min_dim//2)
    start_y = (y//2)-(min_dim//2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

def get_face_region_for_first_frame(frame, previous_box=None):
    detector = MTCNN()
    if previous_box is None:
        detections = detector.detect_faces(frame)
        if detections:
            x, y, w, h = detections[0]['box']
            previous_box = (x, y, w, h)
        else:
            return get_center(frame), None
    else:
        x, y, w, h = previous_box

    face_region = frame[y:y+h, x:x+w]
    return face_region, previous_box

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE), skip_frames=2):
    cap = cv2.VideoCapture(path)
    frames = []
    frame_count = 0
    previous_box = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % skip_frames == 0:
                frame, previous_box = get_face_region_for_first_frame(frame, previous_box)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
                frames.append(frame)
                if len(frames) == max_frames:
                    break
            frame_count += 1
        while len(frames) < max_frames and frames:
            frames.append(frames[-1])
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames, feature_extractor):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)
        frame_mask[i, :length] = 1
        
    return frame_features, frame_mask

def display_frames(frames, title):
    st.subheader(title)
    cols_per_row = 5
    rows = (len(frames) + cols_per_row - 1) // cols_per_row
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            frame_idx = row * cols_per_row + col_idx
            if frame_idx < len(frames):
                with cols[col_idx]:
                    # Convert frame to PIL Image for display
                    frame_pil = Image.fromarray(frames[frame_idx].astype(np.uint8))
                    st.image(frame_pil, caption=f"Frame {frame_idx + 1}", use_container_width =True)#width=120)

def main():
    st.set_page_config(page_title="DeepTrust", page_icon="🕵️")
    st.title("DeepTrust: Detecting Deepfakes")
    st.write("Upload a video to check if it's real or AI-generated")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        try:
            st.video(temp_path)     # preview
            # Load models
            with st.spinner('Loading models...'):
                feature_extractor, lstm_model, class_vocab = load_model()
            # Process video
            with st.spinner('Analyzing video...'):
                frames = load_video(temp_path)
                if len(frames) == 0:
                    st.error("Could not extract frames from the video. Please try a different video.")
                else:
                    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)
                    probabilities = lstm_model.predict([frame_features, frame_mask])[0]
                    st.subheader("Detection Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Resulting probabilities
                        st.write("**Confidence Scores:**")
                        # progress bars
                        for i in np.argsort(probabilities)[::-1]:
                            label = class_vocab[i]
                            prob = probabilities[i] * 100
                            st.write(f"{label.capitalize()}: {prob:.2f}%")
                            st.progress(int(prob))
                    with col2:
                        # Final verdict
                        predicted_class = class_vocab[np.argmax(probabilities)]
                        confidence = np.max(probabilities) * 100
                        st.write("**Final Verdict:**")
                        if predicted_class == 'fake':
                            st.error(f"⚠️ **Warning**: This video is likely manipulated or generated using AI techniques.")
                            st.write(f"Confidence: {confidence:.2f}%")
                        else:
                            st.success(f"✅ **Authentic**: This video appears to be authentic")
                            st.write(f"Confidence: {confidence:.2f}%")

                    st.markdown("---")
                    display_frames(frames, "Frames Used for Analysis")
                    st.info(f"Analysis based on {len(frames)} frames extracted from the video")
        except Exception as e:
            st.error(f"An error occurred while processing the video: {str(e)}")
        finally:
            # Cleaning up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                st.warning(f"Could not clean up temporary file: {str(e)}")

if __name__ == "__main__":
    main()