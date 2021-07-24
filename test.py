import keras
from keras.models import load_model
from keras.applications import InceptionV3
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.models import model_from_json

img_height, img_width = (224,224)
incep_conv = InceptionV3(weights='imagenet',include_top = False, input_shape = (img_height,img_width,3))
x = Flatten()(incep_conv.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)                  
x = Dense(15, activation="softmax")(x)

model = Model(incep_conv.input, x)
model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])

model_json = model.to_json()
with open("inc_model.json", "w") as json_file:
    json_file.write(model_json)

print("DONE")