import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gdown

# Check if model file already exists
if not os.path.exists('trained_model (1).h5'):
    # If not, download it from Google Drive
    url = 'https://drive.google.com/uc?id=13FvRL7p6capnHKtWZn1r4dBSXKmeo1KI'
    output = 'trained_model (1).h5'
    gdown.download(url, output, quiet=False)

@st.cache_resource
def model_prediction(uploaded_file):
    image=tf.keras.preprocessing.image.load_img(uploaded_file,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    with open("label.txt") as f:
        content=f.readlines()
    label=[]
    for i in content:
        label.append(i[:-1])
    index=np.argmax(prediction)
    return label[index]
def load_model():# Load model
    model = tf.keras.models.load_model("trained_model (1).h5")
    return model
model = load_model()# Must be first Streamlit call
st.set_page_config(page_title="Fruit & Veggie Classifier", layout="centered", page_icon="🥦")

# Sidebar Navigation
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📖 About Project", "🔍 Predict"])
# Page: Home
with tab1:
    st.title("🍇 Fruit & Veggie Classifier")
    st.markdown("Welcome to the smart fruit & vegetable detector!")
    image = Image.open("intro.jpg")  # Put the image file in the same directory
    st.image(image, use_column_width=True, caption=None)

# Page: About
with tab2:
    st.title("📖 About the Project")
    st.markdown("""
    This project is a deep learning-based image classifier that can identify different fruits and vegetables 🥕🍎.
    
    - Built using a Convolutional Neural Network (CNN)
    - Trained on a custom dataset of fruits & veggies
    - Powered by Streamlit for a clean and interactive UI
    
    **Technologies Used:**
    - Python 🐍
    - TensorFlow/Keras 🧠
    - Streamlit 🚀
    """)

# Page: Prediction
with tab3:
    st.title("🔍 Upload Image for Prediction")

    uploaded_file = st.file_uploader("Choose a fruit/veggie image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=4,use_column_width=True)
        if (st.button("Predict")):
            st.write("Analyzing image...")
            prediction=model_prediction(uploaded_file)
            st.success(f"✅ Prediction: **{prediction}**")
            st.balloons()
    else:
        st.info("Please upload an image to classify.")
