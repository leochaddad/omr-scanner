from array import array
from flask import Flask, request, jsonify
from process_img import process_img
import numpy as np


app = Flask(__name__)

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
        
        result = process_img(img)
       
        return {"result": result}
      