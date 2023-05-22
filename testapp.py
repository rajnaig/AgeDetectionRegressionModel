# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np

# Load your trained model
from tensorflow.keras.models import load_model

model = load_model('models/age_regression_model.h5')

st.title('Age Prediction App')

# Upload the image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert image to grayscale
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Resize and preprocess the image
    image = image.resize((48, 48))  # Your model expects 48x48 images
    image = np.array(image)
    image = image / 255.0  # Your model expects images scaled in the [0, 1] range
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)  # Add an extra dimension for the grayscale channel

    # Predict age
    age = model.predict(image)

    # Show the predicted age
    st.markdown(f"# Predicted Age: {age}")

