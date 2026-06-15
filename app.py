import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Bidirectional, MaxPool3D, Activation, Dropout, TimeDistributed, Flatten

# --- CONSTANTS & CONFIGURATION ---
IMG_SIZE = 80
NUM_FRAMES = 75  # Standardized sequence length for the video input
CHANNELS = 1     # Grayscale preprocessing to save memory and optimize calculation

# --- VOCABULARY MAPPING (For CTC Decoding) ---
# Standard character-level mapping used in TensorFlow sequence decoding
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", ' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], 
    oov_token=""
)
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# --- VIDEO PREPROCESSING PIPELINE ---
def load_and_preprocess_video(video_path):
    """
    Loads video file, samples frames to a fixed length, converts to grayscale,
    normalizes pixels, and returns a standardized TensorFlow tensor.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    finally:
        cap.release()
        
    frames = np.array(frames)
    
    # Pad or truncate video to match expected NUM_FRAMES
    if len(frames) < NUM_FRAMES:
        padding = np.zeros((NUM_FRAMES - len(frames), IMG_SIZE, IMG_SIZE))
        frames = np.vstack((frames, padding))
    else:
        frames = frames[:NUM_FRAMES]
        
    # Standardize data shape [Frames, Height, Width, Channels] and normalize
    frames = np.expand_dims(frames, axis=-1)
    mean = np.mean(frames)
    std = np.std(frames)
    return tf.cast((frames - mean) / (std + 1e-7), tf.float32)

# --- TENSORFLOW MODEL ARCHITECTURE ---
@st.cache_resource
def load_video_to_text_model():
    """
    Builds a 3D CNN + BiLSTM Spatio-Temporal Model natively in TensorFlow.
    """
    model = Sequential([
        # Spatio-Temporal Feature Extraction (3D CNN)
        Conv3D(16, 3, input_shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS), padding='same'),
        Activation('relu'),
        MaxPool3D((1, 2, 2)),
        
        Conv3D(32, 3, padding='same'),
        Activation('relu'),
        MaxPool3D((1, 2, 2)),
        
        # Reshape / Flatten spatial layout for Sequential LSTM layers
        TimeDistributed(Flatten()),
        
        # Sequence Processing (Bidirectional LSTMs)
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        
        # Dense Character Classification Layer
        Dense(char_to_num.vocabulary_size() + 1, activation='softmax')
    ])
    
    return model

# --- DECODING LAYER ---
def decode_predictions(predictions):
    """
    Uses CTC Greedy Decoder to parse character probabilities back into readable string tokens.
    """
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    # Native TensorFlow CTC Search decoder
    ctc_decode, _ = tf.nn.ctc_greedy_decoder(
        tf.transpose(predictions, perm=[1, 0, 2]), 
        sequence_length=tf.cast(input_len, tf.int32)
    )
    
    # Map back to string format
    dense_tensor = tf.sparse.to_dense(ctc_decode[0]).numpy()
    decoded_text = []
    for row in dense_tensor:
        chars = [num_to_char(val).numpy().decode('utf-8') for val in row if val > 0]
        decoded_text.append("".join(chars))
        
    return decoded_text[0] if decoded_text else "Unable to parse text from video structure."

# --- STREAMLIT UI INTERFACE ---
st.set_page_config(page_title="TF Video-to-Text Studio", layout="centered", page_icon="🎬")

st.title("🎬 Local Video-to-Text Decoder")
st.caption("Powered entirely by local TensorFlow 3D-CNN & Sequence processing architectures. No external API keys required.")

# Initialize the Model instance
with st.spinner("Compiling Neural Network Layers..."):
    model = load_video_to_text_model()

# User inputs: Link download or Direct File upload
source_option = st.radio("Choose Video Source:", ("Upload Local Video File (.mp4)", "Provide Video URL Link"))

video_path = "temp_input_video.mp4"
is_ready = False

if source_option == "Upload Local Video File (.mp4)":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        is_ready = True

else:
    video_url = st.text_input("Paste video URL link here:", placeholder="https://example.com/sample_video.mp4")
    if video_url:
        if st.button("Fetch and Download Video Link"):
            with st.spinner("Downloading video file source locally..."):
                try:
                    # Native Python/Streamlit downloader for clean execution
                    import urllib.request
                    urllib.request.urlretrieve(video_url, video_path)
                    st.success("Video link successfully downloaded!")
                    is_ready = True
                except Exception as e:
                    st.error(f"Error fetching the video link: {e}")

# --- PROCESSING AND PREDICTION PIPELINE ---
if is_ready and os.path.exists(video_path):
    st.video(video_path)
    
    if st.button("Transcribe Video Content", type="primary"):
        with st.status("Processing video frames...", expanded=True) as status:
            
            st.write("🔄 Extracting spatio-temporal matrices and resizing shapes...")
            processed_frames = load_and_preprocess_video(video_path)
            # Expand dimensions to represent Batch Size = 1
            input_tensor = tf.expand_dims(processed_frames, axis=0)
            
            st.write("🧠 Feeding Tensor to TensorFlow 3D-CNN Model architecture...")
            predictions = model.predict(input_tensor)
            
            st.write("📝 Running Connectionist Temporal Classification (CTC) Decoding layer...")
            transcription = decode_predictions(predictions)
            
            status.update(label="Analysis complete!", state="complete", expanded=False)
        
        # Output Interface Display
        st.subheader("📋 Extracted Text Result")
        if transcription.strip() == "":
            # Placeholder display logic if weights are in an untrained/initialized state
            st.info("💡 *Model Architecture executed successfully! To extract meaningful text matching your specific domain language, load a pre-trained `.h5` model weight set matching your vocabulary size using `model.load_weights()`.*")
        else:
            st.success(transcription)
            
    # Cleanup temp storage
    if os.path.exists(video_path):
        os.remove(video_path)
