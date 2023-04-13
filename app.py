from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import numpy as np
from keytotext import pipeline
import os

# Define the Flask application
app = Flask(__name__, template_folder='my_templates')

nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")

# Load the pre-trained model
model = tf.keras.models.load_model('version2.h5')

# Define the image size and maximum sequence length
img_size = (224, 224)
max_length = 32

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32) ##image dtype is Float32
    return x

H=800
W=1200

def generate_caption(img):
    new_size = (800, 1200)
    resized_img = img.resize(new_size)

    # Convert the image to a numpy array
    img_array = np.array(resized_img)

    # Reshape the image array to the required shape
    reshaped_img = img_array.reshape((1, 800, 1200, 3))
    p=model.predict(reshaped_img)
    p1 = np.argmax(p, axis=3)

    # Reshape the predictions array
    p1 = p1.reshape(-1,800, 1200)
    df = pd.read_csv('class_dict_seg.csv')
    
    # Load the cmap array
    cmap = np.array(list(df[[' r', ' g', ' b']].transpose().to_dict('list').values()))

    for i in range(p1.shape[0]):
        # Create a PIL image from the predicted labels using the cmap colors
        predicted_img = Image.fromarray(cmap[p1[i]].astype(np.uint8))
    #predicted_img = Image.fromarray(cmap[p1.shape[0]].astype(np.uint8))
    #cv2.imwrite("pped.jpg",predicted_img)
    img = predicted_img

    uniqueColors = set()
    #img = Image.fromarray(np.uint8(p * 255)).convert('RGB')
    w, h = img.size
    for x in range(w):
        for y in range(h):
            pixel = img.getpixel((x, y))
            uniqueColors.add(pixel)
    
    totalUniqueColors = len(uniqueColors)
    df = df.rename(columns={' r': 'red'})
    df = df.rename(columns={' g': 'green'})
    df = df.rename(columns={' b': 'blue'})
    df['rgb'] = df.apply(lambda x: list([x['red'],x['green'],x['blue']]),axis=1) 
    listOfObjects=[]
    for x in uniqueColors:
        for y in range(len(df)):
            if x[0]==df.rgb[y][0] and x[1]==df.rgb[y][1] and x[2]==df.rgb[y][2]:
                listOfObjects.append(df.name[y])
    if 'unlabeled' in listOfObjects:
        listOfObjects.remove('unlabeled')
    if 'ar-marker' in listOfObjects:
        listOfObjects.remove('ar-marker')
    output_string=nlp(listOfObjects)
   
    print(listOfObjects)
    return output_string


@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file)
        if file:
            # Save the image to the static/images folder
            filename = file.filename
            caption = generate_caption(img)
            # Generate the caption using your generate_caption() function
        
          
    return render_template('index.html', caption=caption, filename=filename)
if __name__ == '__main__':
    app.run(debug=True)
