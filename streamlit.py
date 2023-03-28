import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import numpy as np
from PIL import Image,ImageOps

st.title('CIFAR100 prediction')
key = st.selectbox('Choose image type',('Draw by hand','Upload image'))
model = keras.models.load_model('model\model.h5')
class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','cra','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']


choose = st.sidebar.selectbox(
    "Classes:", class_names)
if choose:
    class_img = Image.open('class_images/'+choose+'.png')
    gray_img = ImageOps.grayscale(class_img)
    st.sidebar.image(gray_img)

if key == 'Upload image':
    uploaded_image = st.file_uploader(label='Image upload',type=['png','jpg'])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()),dtype=np.uint8)
        cv_image = cv2.imdecode(file_bytes,1)
        gray_img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        gray_img = cv2.resize(gray_img,dsize=(32,32))
        gray_img = gray_img.astype('float32')
        gray_img /= 255
        gray_img = np.array([gray_img])
        for_display_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        for_display_image = cv2.resize(for_display_image,dsize=(120,120,))
        st.image(for_display_image)

    with st.form("key1"):
        # ask for input
        button_check = st.form_submit_button("Predict")
    if button_check:
        st.text(class_names[np.argmax(model.predict(gray_img))])
else:
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        update_streamlit=True,
        width = 250,
        height=250,
        drawing_mode='freedraw',
        point_display_radius= 0,
        key="canvas",
    )

    if st.button(label='Predict') and canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img,(32,32))
        img = img.astype('float32')
        img /= 255
        img = np.array([img])
        
        st.text(class_names[np.argmax(model.predict(img))])