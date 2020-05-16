import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re


model_dir = 'model/'
def create_graph():
    """Create the CNN graph"""
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    """Extract bottleneck features"""
    print(list_images)
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []

    create_graph()

    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.

    with tf.compat.v1.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            labels.append('1')

    return features, labels


def init_svm():

    pkl_filename = model_dir + "clf.pkl"
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)
   
  
    return clf

def init_vgg16():
	# json_file = open('models/hemp_classifier_json.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	# #load weights into new model
	# loaded_model.load_weights("models/hemp_classifier_weights.h5")
	#print("Loaded Model from disk")

	# load model
	model = models.load_model(model_dir + "hemp_classifier1.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return model #, graph
