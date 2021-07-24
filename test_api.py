from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
from model_inc import model_load, predict

app = Flask(__name__)
CORS(app)

model = model_load()

def con(image_url):
    imge = Image.open(BytesIO(base64.b64decode(image_url)))
    return imge

@app.route("/api", methods=["POST"])
def home():
    image_base = request.get_json()
    print("img",image_base['cancelled'])
    urii = image_base['uri'].split(",")[1]
    imge = con(urii)
    class_name = predict(model, imge)
    class_name = str(class_name)
    #print(imge.format)
    test_pred_json = {
        "predicted_class": class_name.replace("__"," ").replace("_"," ")
    }
    return jsonify(test_pred_json)
    # return jsonify(image_base)

@app.route("/", methods=["POST", "GET"])
def view():
    if request.method == "POST":
        test_pred = "CLASS 1"
        test_pred_json = {
            "key1": test_pred
        }
        return jsonify(test_pred_json)
    if request.method == "GET":
        return "<h1>Helllo USER from Flask API.</h1>"
    
if __name__ == "__main__":
    app.run(debug=True)