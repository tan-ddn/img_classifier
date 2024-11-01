import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Location Image Classifier")
st.text("Provide URL of Location Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
  # model = tf.keras.models.load_model(r"D:\\ml_projects\\img_classify\\app\\models\\")
  model_vgg16 = tf.keras.applications.vgg16.VGG16()
  model = model_vgg16
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

# classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)  
  img = tf.image.resize(img,[224,224])
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://storage.googleapis.com/image_classification_2021/Glacier-Argentina-South-America-blue-ice.JPEG')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      # label =np.argmax(model.predict(decode_img(content)),axis=1)
      # st.write(classes[label[0]])    
      preds = model.predict(tf.keras.applications.vgg16.preprocess_input(decode_img(content)),axis=1)
      decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(
            preds=preds,
            top=5
        )
      label = decoded_preds[0][0][1]
      st.write(label)
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Image', use_column_width=True)
