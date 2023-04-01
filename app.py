from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('version2.h5')

# Define the Flask application
app = Flask(__name__, template_folder='my_templates')

#app = Flask(__name__)

# Define the image size and maximum sequence length
img_size = (224, 224)
max_length = 32


import pandas as pd
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image





def generate_caption(img):
    

    # Generate the caption
        # Preprocess the image
    #img = preprocess_image(img)
    #img = cv2.imread(img)
    image_buffer = np.frombuffer(img.read(), np.uint8)
    img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    img.shape = (1,800,1200,3)
    p=model.predict(img)
    p1 = np.argmax(p, axis=3)

    # Reshape the predictions array
    p1 = p1.reshape(-1, 800, 1200)
    df = pd.read_csv('class_dict_seg.csv')
      

    # Load the cmap array
    cmap = np.array(list(df[[' r', ' g', ' b']].transpose().to_dict('list').values()))

    # Select an image index to save

    for i in range(p1.shape[0]):
        # Create a PIL image from the predicted labels using the cmap colors
        predicted_img = Image.fromarray(cmap[p1[i]].astype(np.uint8))
    
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
    
    
   

    return listOfObjects




# Define the Flask routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Generate the caption for the image
       # caption = "hello i am prudvish"
        # Generate the caption for the image
        caption = generate_caption(image_file)

        # Render the HTML template with the caption
        return render_template('index.html', caption=caption)

    else:
        # Render the HTML template with the form to upload an image
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
