import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Streamlit page settings
st.set_page_config(
    page_title="Sindhi Alphabet Recognizer",
    page_icon="🔤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Define the custom feature extraction layer
class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
            input_shape=(224, 224, 3),
            trainable=False,
        )

    def call(self, inputs):
        return self.feature_extractor(inputs)

    def get_config(self):
        return super().get_config()

# Register the custom layer
tf.keras.utils.get_custom_objects()["FeatureExtractorLayer"] = FeatureExtractorLayer

# Title
st.markdown("<h2 style='text-align: center;'>🧠 Sindhi Alphabet Recognizer</h2>", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model.keras",
        custom_objects={"FeatureExtractorLayer": FeatureExtractorLayer}
    )

model = load_model()

# Sindhi alphabet class labels
CLASS_NAMES = [
    'ء','ا', 'ب', 'ٻ','ڀ', 'پ','ت','ٺ' ,'ٿ','ٽ', 'ث', 'ج','جھ','ڃ','ڄ','ڇ',
    'چ', 'ح', 'خ','ڊ','ڍ','ڏ','د', 'ذ','ڌ', 'ر', 'ز','ڙ', 'س', 'ش', 'ص',
    'ض','ط', 'ظ', 'ع', 'غ', 'ف','ڦ', 'ق', 'ک','ڪ', 'گ','گھ','ڱ','ڳ', 'ل',
    'م','ن', 'ڻ','ھ','و', 'ي',
]

# File uploader
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Preprocess: Convert to RGB directly (no grayscale), resize, normalize
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Layout: image on left, prediction on right
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="📷 Uploaded", use_column_width=True)
    with col2:
        st.markdown("### 🔍 Prediction")
        st.success(f"**Alphabet:** `{predicted_class}`")
        st.info(f"**Confidence:** `{confidence:.2%}`")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and TensorFlow · Designed for fast, one-page interaction.")
