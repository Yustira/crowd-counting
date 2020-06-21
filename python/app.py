import tensorflow as tf
import numpy as np
import os
import urllib.request
from tensorflow.keras.preprocessing import image
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

upload_folder = "../static/container"

app = Flask(__name__,
            template_folder = "../templates",
            static_folder = "../static")
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = upload_folder
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

extensions = set(["png", "jpg", "jpeg"])

def extension(filename):
	return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

def model(filename):
    model = tf.keras.models.load_model("model_sota.h5")
    src_img = os.path.join(upload_folder, filename)
    img = image.load_img(src_img, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    result = model.predict(img)
    return result
	
@app.route("/")
def upload_form():
	return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
	file = request.files["file"]
	if file.filename == "":
		flash("No image selected for uploading")
		return render_template("home.html")
	if file and extension(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		pred = model(filename)
		return render_template("home.html", 
                                filename = filename,
                                res = ("The number of people in the picture is around " + str(int(round(pred[0,0])))))
	else:
		flash("Allowed image types are : png, jpg, jpeg")
		return render_template("home.html")

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for("static", filename="container/"+filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
