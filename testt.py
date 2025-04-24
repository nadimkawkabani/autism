import streamlit as st
import cv2
import numpy as np
import imghdr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the emotion labels and their corresponding colors and emojis
emotion_labels = {
    'Angry': ('red', '😠'),
    'Disgust': ('green', '🤢'),
    'Fear': ('purple', '😨'),
    'Happy': ('blue', '😄'),
    'Sad': ('gray', '😔'),
    'Surprise': ('orange', '😲'),
    'Neutral': ('black', '😐')
}

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale and resize it to 48x48 pixels
    resized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(resized, (48, 48))
    # Convert the grayscale image to a NumPy array
    img_array = img_to_array(resized)
    # Add a dimension to the array to account for batch size (required by the model)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0
    return img_array

# Load the saved model
model = load_model('model.h5')

# Set page configuration and styling
st.set_page_config(page_title='Facial Expression Recognition App - MSBA315', page_icon=':smiley:', layout='wide')
st.write('<style>body{background-color: #f5f5f5;}</style>', unsafe_allow_html=True)

# Define the Streamlit app
def main():
    st.title('MSBA315: Machine Learning and Predictive Analytics')
    st.title('Facial Expression Recognition App')
    st.header('Joe Sayegh - Tarek Riman - Eslam Abo Al Hawa - Shadi Youssef')
    # Upload an image file
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Check if the file is an image
            if imghdr.what(uploaded_file) is None:
                raise Exception('Invalid file format or file is not an image')

            # Load the image
            img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            # Preprocess the image
            preprocessed_img = preprocess_image(img)
            # Make a prediction using the loaded model
            prediction = model.predict(preprocessed_img)
            # Get the predicted emotion label
            predicted_label = np.argmax(prediction)
            # Show the image and the predicted emotion label
            color = emotion_labels[list(emotion_labels.keys())[predicted_label]][0]
            emoji = emotion_labels[list(emotion_labels.keys())[predicted_label]][1]
            st.image(uploaded_file, caption='', width=350)
            st.write(f'<h2 style="color: {color};">Predicted Emotion: {list(emotion_labels.keys())[predicted_label]} {emoji}</h2>', unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

if __name__ == '__main__':
    main()

