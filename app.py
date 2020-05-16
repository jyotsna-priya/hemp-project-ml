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
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.image as mpimg
from skimage.transform import resize

UPLOAD_FOLDER = os.path.join('static', 'images')
FILEPATH = "static/images/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#img = ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_images():
    for root, dirs, files in os.walk(FILEPATH):
        for file in files:
            os.remove(os.path.join(root, file))

def image_read_scaled(path, size=(128,128)):
    img = mpimg.imread(path)
    img = resize(img, size, anti_aliasing=True)
    return img

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
            img = user_image
        return render_template('options.html')

#route for options
@app.route('/options', methods = ['GET'])
def show_options():
    return render_template('options.html')

#route for svm
@app.route('/svm', methods = ['GET'])
def svm():

    model = init_svm()
    print ("svm model")
    test_subset_data_dir = "static/images/"
    #print ("image" +file)
    list_images = []
    #for image in glob.glob('static/images/*.jpg'):
    for filename in os.listdir(test_subset_data_dir):#assuming jpg
        image= test_subset_data_dir+filename
        print(image)
        #img=image_read_scaled(image)
        list_images.append(image)
        #print (image)
    #list_images.append(file)
    print ('I amout of for loop')
    x_test_data, y_test_data = extract_features(list_images)

    y_pred = model.predict(x_test_data)
    prob = model.predict_proba(x_test_data)
    max_prob = np.amax(prob) * 100.0
    perc = ("{0:.2f}".format(max_prob))
   
    print('y_pred2', y_pred)
    # print(category_names)
    # print('test_subset_generator.classes', test_subset_generator.classes)

    #x, y = test_subset_generator.next()
    img_nr = 0
    category_names = ['HealthyHemp', 'HempBudRot', 'HempLeafSpot', 'HempNutrientDef', 'HempPowderyMildew', 'NonHemp']
	#pred_emotion = category_names[y_pred2[img_nr]]
    if y_pred == ['1']:
        disease_name = 'Healthy Hemp'
    elif y_pred == ['4']:
        disease_name = 'HempBudRot'
    elif y_pred == ['2']:
        disease_name = 'Hemp Leaf Spot'
    elif y_pred == ['3']:
        disease_name = 'Hemp Nutrient Deficiency'
    elif y_pred == ['5']:
        disease_name = 'Hemp Powdery Mildew'
    elif y_pred == ['0']:
        disease_name = 'Non Hemp'
    else:
        disease_name = 'Model Not Trained On such Image'

    fileName = os.listdir(FILEPATH)[0]
    user_image = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
	#print('user image', user_image)

    return render_template('predict.html', disease_name = disease_name, 
                           user_image = user_image, perc = perc)



@app.route('/vgg16', methods = ['GET'])
def vgg16():
	model = init_vgg16()

	test_subset_data_dir = "static/"

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

	fileName = os.listdir(FILEPATH)[0]
	user_image = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
	#print('user image', user_image)

	return render_template('predict.html', pred_emotion = pred_emotion, disease_name = disease_name, user_image = user_image, perc = perc)



if __name__ == '__main__':
    app.run(debug=True,
            host='127.0.0.1',
            port=5555)
