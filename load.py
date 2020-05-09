#import numpy as np
#import keras.models
# from keras.models import model_from_json
#from scipy.misc import imread, imresize,imshow
from tensorflow.keras import layers, models, Model, optimizers
import tensorflow as tf

def init():
	# json_file = open('models/hemp_classifier_json.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	# #load weights into new model
	# loaded_model.load_weights("models/hemp_classifier_weights.h5")
	#print("Loaded Model from disk")

	# load model
	model = models.load_model("/Users/jyotsna/Sites/cmpe257/models/hemp_classifier1.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	# graph = tf.get_default_graph()

	return model #, graph