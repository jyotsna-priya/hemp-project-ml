from flask import Flask, render_template, request
import json
import requests
from datetime import datetime
import pytz
import os, re, os.path
from flask import flash, redirect #url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from load import *
import numpy as np
import glob

UPLOAD_FOLDER = os.path.join('static', 'images')
FILEPATH = "/Users/jyotsna/Sites/cmpe257/static/images/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_images():
    for root, dirs, files in os.walk(FILEPATH):
        for file in files:
            os.remove(os.path.join(root, file))

# routes for hemp app
@app.route('/hemp', methods = ['GET', 'POST'])
def hemp_func():
    if request.method == 'GET':
        delete_images()
        return render_template('hemp.html')

    if request.method == 'POST':
        print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        return render_template('options.html')

#route for options
@app.route('/options', methods = ['GET'])
def show_options():
	return render_template('options.html')

#route for vgg16
@app.route('/vgg16', methods = ['GET'])
def vgg16():
	model = init_vgg16()

	test_subset_data_dir = "/Users/jyotsna/Sites/cmpe257/static/"

	test_datagen =  ImageDataGenerator(
	    rescale=1./255
	)

	#print('Total number of images for "final testing":')
	test_subset_generator = test_datagen.flow_from_directory(
	test_subset_data_dir,
	batch_size = 1,
	target_size = (224, 224),
	class_mode = "categorical",
	shuffle=False)

	Y_pred2 = model.predict_generator(test_subset_generator, 1, verbose=1)
	prob = Y_pred2[0]
	perc = round((100 * max(prob)), 2)
	y_pred2 = np.argmax(Y_pred2, axis=1)
	print(max(prob))

	print('y_pred2', y_pred2)
	# print(category_names)
	# print('test_subset_generator.classes', test_subset_generator.classes)

	x, y = test_subset_generator.next()
	img_nr = 0
	category_names = ['HealthyHemp', 'HempBudRot', 'HempLeafSpot', 'HempNutrientDef', 'HempPowderyMildew', 'NonHemp']
	pred_emotion = category_names[y_pred2[img_nr]]
	if pred_emotion == 'HealthyHemp':
	    disease_name = 'Healthy Hemp'
	elif pred_emotion == 'HempBudRot':
	    disease_name = 'HempBudRot'
	elif pred_emotion == 'HempLeafSpot':
	    disease_name = 'Hemp Leaf Spot'
	elif pred_emotion == 'HempNutrientDef':
	    disease_name = 'Hemp Nutrient Deficiency'
	elif pred_emotion == 'HempPowderyMildew':
	    disease_name = 'Hemp Powdery Mildew'
	elif pred_emotion == 'NonHemp':
	    disease_name = 'Non Hemp'
	else:
	    disease_name = 'Model Not Trained On such Image'

	#user_image = os.path.join(app.config['UPLOAD_FOLDER'], str(os.listdir(app.config['UPLOAD_FOLDER'])).strip('[]'))
	#user_image = os.listdir(app.config['UPLOAD_FOLDER'])
	fileName = os.listdir(FILEPATH)[0]
	user_image = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
	print('user image', user_image)

	return render_template('predict.html', pred_emotion = pred_emotion, disease_name = disease_name, user_image = user_image, perc = perc)


if __name__ == '__main__':
    app.run(debug=True,
            host='127.0.0.1',
            port=5555)
