import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS Styling
# --------------------------------------------------

st.markdown("""
<style>

body {
background: linear-gradient(120deg,#89f7fe,#66a6ff);
}

.main-title{
font-size:50px;
font-weight:800;
text-align:center;
color:#ffffff;
text-shadow:2px 2px 10px rgba(0,0,0,0.3);
margin-bottom:10px;
}

.subtitle{
font-size:22px;
text-align:center;
color:white;
margin-bottom:40px;
}

.upload-box{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 5px 20px rgba(0,0,0,0.15);
}

.prediction-card{
background:linear-gradient(135deg,#ff9a9e,#fad0c4);
padding:25px;
border-radius:20px;
text-align:center;
box-shadow:0px 8px 25px rgba(0,0,0,0.25);
transition:0.3s;
}

.prediction-card:hover{
transform:scale(1.05);
}

.pred-digit{
font-size:60px;
font-weight:900;
color:#2c2c2c;
}

.confidence{
font-size:22px;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.h5")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown('<div class="main-title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Model trained on MNIST Dataset</div>', unsafe_allow_html=True)

# --------------------------------------------------
# File Upload
# --------------------------------------------------

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png","jpg","jpeg"]
)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("L")
    image = image.resize((28,28))

    img_array = np.array(image)

    img_array = 255 - img_array
    img_array = img_array / 255.0

    img_array = img_array.reshape(1,28,28)

    return img_array,image

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    processed_image,display_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)*100)

    col1,col2 = st.columns(2)

    with col1:
        st.image(display_image,width=250,caption="Uploaded Image")

    with col2:

        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        st.markdown("### Predicted Digit")

        st.markdown(f'<div class="pred-digit">{digit}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

        st.progress(int(confidence))

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # Probability distribution
    st.subheader("Prediction Probability Distribution")

    prob_df = pd.DataFrame(
        prediction[0],
        columns=["Probability"],
        index=[0,1,2,3,4,5,6,7,8,9]
    )

    st.bar_chart(prob_df)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""
<hr>

<center style="font-size:18px;">
Built with ❤️ using TensorFlow & Streamlit
</center>
""", unsafe_allow_html=True)