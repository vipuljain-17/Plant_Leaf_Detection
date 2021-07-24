from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
from model import model_load, predict

app = Flask(__name__)
CORS(app)

model = model_load()

def con(image_url):
    # img = base64.b64decode(image_url)
    imge = Image.open(BytesIO(base64.b64decode(image_url)))
    return imge

@app.route("/api", methods=["POST"])
def home():
    image_base = request.get_json()
    # img = con(image_base)
    print("img",image_base['cancelled'])
    # test_pred = img.mode
    # test_pred_json = {
    #     "key1": test_pred
    # }
    urii = image_base['uri'].split(",")[1]
    imge = con(urii)
    class_name = str(predict(model,imge)).replace("__"," ").replace("_"," ")
    #print(imge.format)
    test_pred_json = {
        "key1": class_name
    }
    return jsonify(test_pred_json)
    # return jsonify(image_base)

@app.route("/", methods=["POST"])
def view():
    test_pred = "CLASS 1"
    test_pred_json = {
        "key1": test_pred
    }
    return jsonify(test_pred_json)
    
if __name__ == "__main__":
    app.run(debug=True)