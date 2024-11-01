import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
# import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Image Classifier")
st.text("Provide URL of Image for image classification")

@st.cache_resource
def load_model():
  # model = tf.keras.models.load_model(r"D:\\ml_projects\\img_classify\\app\\models\\")
  model = tf.keras.applications.inception_v3.InceptionV3()
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

# classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)  
  img = tf.image.resize(img,[299,299])
  # return np.expand_dims(img, axis=0)
  return tf.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      # label =np.argmax(model.predict(decode_img(content)),axis=1)
      # st.write(classes[label[0]])    
      preprocess_input = tf.keras.applications.inception_v3.preprocess_input
      decoded_img = decode_img(content)
      decoded_img = preprocess_input(decoded_img)
      preds = model.predict(decoded_img)
      decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(
            preds=preds,
            top=5
        )
      label = decoded_preds[0][0][1]
      score = decoded_preds[0][0][2] * 100
      prediction = label + ' ' + str('{:.2f}%'.format(score))
      print(prediction)
      st.write(prediction)
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Image', use_column_width=True)
