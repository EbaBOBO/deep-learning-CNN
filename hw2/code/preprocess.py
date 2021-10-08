import pickle
import numpy as np
import tensorflow as tf
import os

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	# TODO: Do the rest of preprocessing! 
	input1=[]
	label1=[]
	for i in range(len(labels)):
		if(labels[i]==first_class or labels[i]==second_class):
			input1.append(inputs[i])
		else:
			labels[i]=0
	labels=np.array(labels)
	label1=labels[np.nonzero(labels)]
	inputs=tf.reshape(input1, (-1, 3, 32 ,32))
	inputs=tf.transpose(inputs, perm=[0,2,3,1])
	inputs=np.float32(inputs)/255
	label2=[]
	for i in range(len(label1)):
		if(label1[i]==second_class):
			label1[i]=1
		else:
			label1[i]=0
	label2=tf.one_hot(label1,2)
	# print(inputs.shape) (10000,32,32,3)
	# print(label2.shape) (10000,2)
	return inputs,label2

# get_data('/Users/zccc/1470projects/project2/data/train',3,5)