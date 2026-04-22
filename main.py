# Packages: streamlit, tensorflow, opencv-python, openai, python-dotenv, pillow

import cv2 # used for image resizing and basic image processing
import numpy as np # helps handle images as arrays (matrices of numbers)
import streamlit as st # builds the web app interface
import os

# MobileNetV2: a pretrained deep learning model (trained on ImageNet dataset)
# preprocess_input: prepares images in the correct format for MobileNetV2
# decode_predictions: converts model output numbers into human-readable labels
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image # opens and reads uploaded images
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Loads the MobileNetV2 neural network
def load_model():
    model = MobileNetV2(weights="imagenet") # weights="imagenet": means it already knows 1000 common objects (cat, cars, dogs, etc.)
    return model

# Prepares the image so that AI model can understand it
def preprocess_image(image):
    img = np.array(image) # converts image into numbers (pixels)
    img = cv2.resize(img, (224, 224)) # resizes image to 224x224 (required by MobileNetV2)
    img = preprocess_input(img) # normalizes pixel values (important for AI accuracy)
    img = np.expand_dims(img, axis=0) # adds batch dimension
    return img

# Makes predictions using the model
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image) # converts raw image into AI-ready format
        predictions = model.predict(processed_image) # runs the image through the neutral network; outputs probability scores for 1000 classes
        decoded_predictions = decode_predictions(predictions, top=3)[0] # converts raw probabilities into readable labels; returns top 3 guesses
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

# Make a recipe
def get_recipe(food_name):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
        The detected food is: {food_name}.
        
        Please provide:
        1. A short description of the dish
        2. Ingredients list
        3. Step-by-step cooking instructions
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional chef."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error getting recipe: {str(e)}"
    
def main():
    # Brower tab title
    st.set_page_config(page_title="My AI Kitchen", page_icon="🌮", layout="centered")
    
    # UI Text
    st.write("By My Lam 👩🏻‍💻")
    st.title("My AI Kitchen 🌮")
    st.write("Upload an image or type the food to get the recipe! 🥖")
    
    # Saves model in memory so it does not reload every time -> Makes app faster
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    # Loads the model once
    model = load_cached_model()
    
    # Lets user upload an image, only accepts JPG or PNG files
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    # Or type the food name
    food_input = st.text_input(
        "Or type a food name (optional)", 
        disabled=uploaded_file is not None
    )

    # Show warming if nothing provided
    if uploaded_file is None and not food_input:
        st.info("Please upload an image or type a food name.")
    
    # Case 1: Image uploaded
    if uploaded_file is not None:
        # Shows the uploaded image in the app
        image = st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # When the button is clicked
        if st.button("Classify Image"):
            # Shows loading animation
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file) # opens image properly using PIL
                predictions = classify_image(model, image) # sends image properly using PIL
                
                # Show Results
                if predictions:
                    top_label = predictions[0][1] # best prediction

                    st.markdown(f"##### Predictions: **{top_label}**") # displays results section
                    st.write(f"({predictions[0][2]:.2%})")

                    # Check the Possibilities
                    # st.subheader("Other Possibilities")
                    # for _, label, score in predictions[1:]:
                    #     st.write(f"{label}: {score:.2%}")

                    # Get recipe from OpenAI
                    st.subheader(f"🍳 Recipe")
                    with st.spinner("Generating recipe..."):
                        recipe = get_recipe(top_label)
                        st.write(recipe)

    # Case 2: No image, text input only
    elif food_input:
        # When the button is clicked
        if st.button("Classify Food"):
            st.subheader(f"🍳 Recipe for {food_input}")
            with st.spinner("Generating recipe..."):
                recipe = get_recipe(food_input)
                st.write(recipe)
                        
if __name__ == "__main__":
    main()
