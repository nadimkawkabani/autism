
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import imghdr

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

# Load the saved model
model = load_model('model.h5')

# Set page configuration and styling
st.set_page_config(page_title='Facial Expression Recognition App - MSBA315', page_icon=':smiley:', layout='wide')
st.write('<style>body{background-color: #f5f5f5;}</style>', unsafe_allow_html=True)

# Define the Streamlit app
def main():
    st.title('MSBA315: Machine Learning and Predictive Analytics')
    st.title('Using Machine Learning to Identify Autism:  An Analysis of Predictive Models and Facial Expression Classification APP ')
    st.header('Amira - Dania - Nadim - Yasmina')
    
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            if imghdr.what(uploaded_file) is None:
                raise Exception('Invalid file format or file is not an image')

            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            image = image.resize((48, 48))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction)

            color = emotion_labels[list(emotion_labels.keys())[predicted_label]][0]
            emoji = emotion_labels[list(emotion_labels.keys())[predicted_label]][1]

            st.image(uploaded_file, caption='', width=350)
            st.write(f'<h2 style="color: {color};">Predicted Emotion: {list(emotion_labels.keys())[predicted_label]} {emoji}</h2>', unsafe_allow_html=True)

        except Exception as e:
            st.error(str(e))

if __name__ == '__main__':
    main()
