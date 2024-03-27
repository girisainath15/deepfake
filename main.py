import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title('Deepfake Image Detector')

# Load the saved model
# @st.cache(allow_output_mutation=True)
def load_model():
    print("111111111111111111")
    model = tf.keras.models.load_model('test.h5')
    print("22222222222222222222")
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize to match Xception input shape
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.xception.preprocess_input(image)
    return image

# Function to make predictions
def predict_image(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

# Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(image)
        if prediction[0][0] > 0.2:
            st.write('This image is classified as a deepfake.')
        else:
            st.write('This image is classified as real.')
