import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

st.write("""
# Pneumonia Predictor
The training of the model and development of this app is done by Tanuj Jain.
\nThis model is trained on VGG19 model using the Chest X ray dataset.

""")

# Load the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Convert the image to an array
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array=np.expand_dims(image_array,axis=0)
    image_array /= 255.0 #normailze pixel values
    # Load the model
    model = load_model('vgg19_trained_model.keras')
    # Make a prediction
    
    prediction = model.predict(image_array)
    if prediction[0] > 0.5:
        st.subheader("Pneumonia Detected")
    else:
        st.subheader("Normal")