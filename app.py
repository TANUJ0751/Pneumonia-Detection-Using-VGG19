import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
st.write("""
# Pneumonia Predictor
The training of the model and development of this app is done by Tanuj Jain.
\nThis model is trained on VGG19 model using the Chest X ray dataset.

""")
file_id ="1-HQyY4P519xvcTGv-IskvNSQVmL9NkHs"
url = f'https://drive.google.com/uc?export=download&id={file_id}'
def load_model_from_drive(url):
    # Download the model from Google Drive using gdown
    output_path = './model.keras'  # Temporary path to store the model file
    gdown.download(url, output_path, quiet=False)

    # Load the model from the file
    model = tf.keras.models.load_model(output_path)
    return model
try:
    model = load_model_from_drive(url)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
# Load the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Convert the image to an array
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array=np.expand_dims(image_array,axis=0)
    image_array /= 255.0 #normailze pixel values
    # Load the model
    
    # Make a prediction
    
    prediction = model.predict(image_array)
    if prediction[0] > 0.5:
        st.subheader("Pneumonia Detected")
    else:
        st.subheader("Normal")