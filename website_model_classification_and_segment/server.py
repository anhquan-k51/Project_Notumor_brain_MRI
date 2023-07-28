from flask import Flask, render_template, request,redirect, send_file
from keras import models
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import random
from tensorflow.keras.utils import load_img
import string
from flask_cors import CORS
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate 
import skimage.transform as trans
import keras 
from PIL import ImageOps
from IPython.display import Image, display

def unet_bt(pretrained_weights=None, input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides = 2)(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides = 2)(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides = 2)(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides = 2)(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    
    return model

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'predict'

model = models.load_model("./model/modelvgg16.h5")
segModel = unet_bt('model/weights.hdf5')
# testModel = unet_bt()
# testModel.save('tttt')
# def find_bbox():

def segImage(image_path):
    org = cv2.imread(image_path)
    org = np.asarray(org)
    rows, cols, channels = org.shape     

    img = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = trans.resize(img,(256,256))

    predicted = segModel.predict(np.reshape(img, (1, 256, 256, 1)))
    predicted = predicted.astype(np.float64) * 255
    predicted = np.reshape(predicted, (256, 256))
    predicted = trans.resize(predicted, (rows,cols))
    predicted = predicted.astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)

    ret, mask = cv2.threshold(predicted, 120, 255, cv2.THRESH_BINARY)
    white_pixels = np.where((mask[:, :, 0] == 255) & 
                            (mask[:, :, 1] == 255) & 
                            (mask[:, :, 2] == 255))
    
    mask[white_pixels] = [0, 0, 255]

    masked = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert image to RGB color space

    lower_red = np.array([0, 50, 50])      # Lower bound for red color
    upper_red = np.array([10, 255, 255])   # Upper bound for red color

    hsv_image = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add length and width text to the bounding box
        width_text = f"Width: {round(w*0.2645833333,2)}x{round(h*0.2645833333,2)}"
        cv2.putText(org, width_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    add = cv2.addWeighted(org, 0.9, mask, 0.7, 0)  

    return add

def map_predict(arr):
    map_class = ['glioma', 'meningioma', 'notumor', 'pituitary']
    max_value = max(arr)  # Find the maximum value in the array
    max_index = np.argmax(arr)  # Find the index of the maximum value
    return map_class[max_index], max_value

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict-seg")
def predict_seg():
    image_path = 'test/y1.jpg'
    
    org = cv2.imread(image_path)
    org = np.asarray(org)
    rows, cols, channels = org.shape     

    img = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = trans.resize(img,(256,256))

    predicted = segModel.predict(np.reshape(img, (1, 256, 256, 1)))
    predicted = predicted.astype(np.float64) * 255
    predicted = np.reshape(predicted, (256, 256))
    predicted = trans.resize(predicted, (rows,cols))
    predicted = predicted.astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)

    ret, mask = cv2.threshold(predicted, 120, 255, cv2.THRESH_BINARY)
    white_pixels = np.where((mask[:, :, 0] == 255) & 
                            (mask[:, :, 1] == 255) & 
                            (mask[:, :, 2] == 255))
    mask[white_pixels] = [0, 0, 255]
    add = cv2.addWeighted(org, 0.9, mask, 0.7, 0)   

    cv2.imshow('image', cv2.resize(mask,(500,500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render_template('index.html')
    

@app.route("/img")
def img():
    return send_file(request.args.get('filename'), mimetype='image/jpg')

@app.route("/predict",methods=['post'])
def predict():
    class_result = ''
    prob = ''
    if request.method == 'POST':
      f = request.files['file']
      if f.filename == '':
          return redirect('')
      fileName = get_random_string(20)+'.jpg'
      pathFile = 'predict/'+fileName
      f.save(pathFile)

      img = image.load_img(pathFile, target_size=(224, 224))
      img_array = np.expand_dims(img, axis=0)
      result = model.predict(img_array)
      class_result,prob = map_predict(result[0])

      img_seg = segImage(pathFile)
      fileName = get_random_string(20)+'.jpg'
      pathFile = 'predict/'+fileName
      cv2.imwrite(pathFile, cv2.resize(img_seg,(500,500)))
      print(class_result,prob)

    return render_template('index.html',class_predict=class_result,prob=round(prob*100, 2),pathFile=pathFile)

if __name__ == '__main__':
  app.run(debug=True)