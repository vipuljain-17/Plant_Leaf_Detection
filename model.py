import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
import PIL
from PIL import Image
import numpy as np

def model_load():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("vgg16_1.h5")
    model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=["accuracy"])
    return model

def predict(model,img):
    img_height, img_width = (224,224)
    img = img.resize((img_height,img_width), Image.ANTIALIAS)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    class_list=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
	'Potato___Early_blight','Potato___healthy','Potato___Late_blight',
	'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy',
	'Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
	'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',
	'Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus']

    class_id = class_list[np.argmax(model.predict(img))]
    return class_id