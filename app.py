from array import array
from flask import Flask, request, jsonify
from process_img import process_img
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# should get an image sent in the body of the request
# and return the processed image

@app.route("/process", methods=["POST"])
def process():
    if request.method == "POST":
        # get the image from the request
        print("request received")
        print(request.files)
        img = request.files["image"]
        
        # convert the image to a numpy array
        img = np.frombuffer(img.read(), dtype=np.uint8)
        
        image_b64, answers  = process_img(img)
       
        return {"image_b64": image_b64, "answers": answers }
      