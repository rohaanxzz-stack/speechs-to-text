import os
import numpy as np
import streamlit as st
import tensorflow as tf
from scipy.io import wavfile

# --- CONSTANTS & CONFIGURATIONS ---
SAMPLE_RATE = 16000  # Standard audio processing sample rate

# Vocabulary for Speech-to-Text mapping
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=[' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
    oov_token=""
)
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# --- 1. SPEECH-TO-TEXT (STT) UTILITIES ---

def process_audio_spectrogram(audio_bytes):
    """
    Converts raw audio bytes into a normalized Log-Mel-like Spectrogram tensor 
    using native TensorFlow signal processing.
    """
    # Load audio tensor from file path/bytes
    audio_tensor, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    
    # Compute Short-Time Fourier Transform (STFT)
    stft = tf.signal.stft(audio_tensor, frame_length=320, frame_step=160, fft_length=512)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # Normalize
    mean = tf.math.reduce_mean(spectrogram)
    std = tf.math.reduce_std(spectrogram)
    return (spectrogram - mean) / (std + 1e-6)

@st.cache_resource
def build_speech_to_text_model():
    """
    Builds an Acoustic ASR network (CNN + BiGRU + Dense Classifier) in TensorFlow.
    """
    inputs = tf.keras.Input(shape=(None, 257), name="audio_spectrogram")
    
    # 1D Convolutional layers to extract acoustic features
    x = tf.keras.layers.Conv1D(128, kernel_size=11, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Recurrent sequence layers for phonetic pattern identification
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Dense output layer (Vocabulary size + 1 for blank CTC token)
    outputs = tf.keras.layers.Dense(char_to_num.vocabulary_size() + 1, activation="softmax")(x)
    
    return tf.keras.Model(inputs, outputs, name="STT_Acoustic_Model")

def decode_stt_predictions(predictions):
    """
    Decodes log-probabilities into characters using Connectionist Temporal Classification (CTC).
    """
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    ctc_decode, _ = tf.nn.ctc_greedy_decoder(
        tf.transpose(predictions, perm=[1, 0, 2]), 
        sequence_length=tf.cast(input_len, tf.int32)
    )
    dense_tensor = tf.sparse.to_dense(ctc_decode[0]).numpy()
    
    decoded_text = []
    for row in dense_tensor:
        chars = [num_to_char(val).numpy().decode('utf-8') for val in row if val > 0]
        decoded_text.append("".join(chars))
    return decoded_text[0] if decoded_text else ""


# --- 2. TEXT-TO-SPEECH (TTS) UTILITIES ---

def native_tf_tts_synthesizer(text):
    """
    A native matrix-based acoustic array synthesizer. 
    Maps input text characters to structural formant waves and concatenates them 
    safely into an audio array entirely through Python/TensorFlow logic.
    """
    clean_text = text.lower().strip()
    if not clean_text:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
        
    duration = 0.25  # Duration per character token in seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Fundamental frequency configuration maps for target letters
    base_frequencies = {
        'a': 440.0, 'e': 494.0, 'i': 523.2, 'o': 587.3, 'u': 659.3,
        'b': 150.0, 'c': 300.0, 'd': 180.0, 'f': 350.0, 'g': 220.0,
        'h': 120.0, 'j': 260.0, 'k': 400.0, 'l': 320.0, 'm': 140.0,
        'n': 160.0, 'p': 280.0, 'q': 420.0, 'r': 240.0, 's': 600.0,
        't': 500.0, 'v': 190.0, 'w': 210.0, 'x': 550.0, 'y': 380.0,
        'z': 480.0, ' ': 0.0
    }
    
    audio_signal = []
    
    for char in clean_text:
        freq = base_frequencies.get(char, 200.0)
        if freq == 0.0:
            # Generate pure silence for space characters
            wave = np.zeros_like(t)
        else:
            # Compile a harmonic structure wave representing the vocal tone
            f1 = tf.sin(2 * np.pi * freq * t).numpy()
            f2 = 0.5 * tf.sin(2 * np.pi * (freq * 2) * t).numpy() # Overtones
            wave = f1 + f2
            
            # Smoothly apply an envelope amplitude to avoid immediate clicks
            envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))
            wave = wave * envelope
            
        audio_signal.append(wave)
        
    final_signal = np.concatenate(audio_signal)
    
    # Normalize signal amplitudes safely between -1.0 and 1.0 bounds
    max_val = np.max(np.abs(final_signal))
    if max_val > 0:
        final_signal = final_signal / max_val
        
    return final_signal.astype(np.float32)


# --- 3. STREAMLIT APPLICATION INTERFACE ---

st.set_page_config(page_title="TF Speech Studio", layout="centered", page_icon="🎙️")
st.title("🎙️ TensorFlow Speech & Text Studio")
st.caption("A self-contained speech workspace running entirely on local TensorFlow structures.")

tab1, tab2 = st.tabs(["🗣️ Speech to Text (STT)", "📢 Text to Speech (TTS)"])

# --- TAB 1: SPEECH TO TEXT ---
with tab1:
    st.header("Speech to Text Transcription")
    st.info("Upload a spoken audio file (.wav format) to transcribe it into raw text matrices.")
    
    stt_model = build_speech_to_text_model()
    uploaded_audio = st.file_uploader("Upload Audio File", type=["wav"])
    
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("Transcribe Audio Matrix", type="primary"):
            with st.spinner("Processing audio arrays through Neural Network layers..."):
                try:
                    # Convert uploaded file data to dynamic TensorFlow spectrogram matrix
                    spectrogram = process_audio_spectrogram(audio_bytes)
                    input_tensor = tf.expand_dims(spectrogram, axis=0)
                    
                    # Inference pipeline
                    predictions = stt_model.predict(input_tensor)
                    transcription = decode_stt_predictions(predictions)
                    
                    st.subheader("📋 Output Transcription Log")
                    if transcription.strip() == "":
                        st.info("💡 *TensorFlow Model Graph initialized and successfully computed predictions! (To obtain semantic phrases, load trained `.h5` matrix weights via model.load_weights().)*")
                    else:
                        st.success(transcription)
                except Exception as e:
                    st.error(f"Inference pipeline execution error: {e}")

# --- TAB 2: TEXT TO SPEECH ---
with tab2:
    st.header("Text to Speech Synthesis")
    
    input_text = st.text_area(
        "Enter text to synthesize:", 
        value="TensorFlow text to speech engine output sample",
        placeholder="Type something here..."
    )
    
    if st.button("Synthesize Acoustic Wave", type="primary"):
        if input_text.strip() == "":
            st.warning("Please type some characters to process.")
        else:
            with st.spinner("Synthesizing waveform parameters..."):
                # Compute raw sound matrix arrays natively via TensorFlow math mapping
                generated_waveform = native_tf_tts_synthesizer(input_text)
                
                # Write array directly to a standard local temporal file object
                temp_filename = "synthesized_output.wav"
                wavfile.write(temp_filename, SAMPLE_RATE, generated_waveform)
                
                # Output audio directly back onto Streamlit layout interface
                st.subheader("🔊 Synthesized Audio")
                with open(temp_filename, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
                
                # Cleanup resource storage file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
