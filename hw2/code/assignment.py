from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.W1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
        self.W3 = tf.Variable(tf.random.truncated_normal([3,3,20,20], stddev=0.1)) 
        self.Wdense1 = tf.Variable(tf.random.truncated_normal([3*3*20,80], stddev=0.1))     
        self.Wdense2 = tf.Variable(tf.random.truncated_normal([80,20], stddev=0.1)) 
        self.Wdense3 = tf.Variable(tf.random.truncated_normal([20,2], stddev=0.1))         
        self.b1 = tf.Variable(tf.random.normal(shape=[16], stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.normal(shape=[20], stddev=0.1, dtype=tf.float32))
        self.b3 = tf.Variable(tf.random.normal(shape=[20], stddev=0.1, dtype=tf.float32))
        self.bdense1 = tf.Variable(tf.random.truncated_normal([80], stddev=0.1))     
        self.bdense2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1)) 
        self.bdense3 = tf.Variable(tf.random.truncated_normal([2], stddev=0.1))
        # TODO: Initialize all trainable parameters

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
#layer1
        layer1c=tf.nn.conv2d(inputs, self.W1, strides=[1, 2, 2, 1], padding='SAME')
        #(100,16,16,16)
        layer1b=tf.nn.bias_add(layer1c, self.b1)
        mean1,variance1=tf.nn.moments(layer1b, [0,1,2])
        layer1n=tf.nn.batch_normalization(layer1b, mean1, variance1,offset=None,scale=None,variance_epsilon=1e-5)
        layer1r=tf.nn.relu(layer1n)
        layer1m=tf.nn.max_pool(layer1r, ksize=3, strides=2, padding="VALID")
        #(100,7,7,16)

#layer2
        layer2c = tf.nn.conv2d(layer1m, self.W2, strides=[1, 1, 1, 1], padding='SAME')
        #(100,7,7,20)
        layer2b=tf.nn.bias_add(layer2c, self.b2)
        mean2,variance2=tf.nn.moments(layer2b, [0,1,2])
        layer2n=tf.nn.batch_normalization(layer2b, mean2, variance2,offset=None,scale=None,variance_epsilon=1e-5)
        layer2r=tf.nn.relu(layer2n)
        layer2m=tf.nn.max_pool(layer2r, ksize=2, strides=2, padding="VALID")
        #(100,3,3,20)

#layer3
        layer3c=tf.nn.conv2d(layer2m, self.W3, strides=[1, 1, 1, 1], padding='SAME')
        #(100,3,3,20)
        layer3b=tf.nn.bias_add(layer3c, self.b3)
        mean3,variance3=tf.nn.moments(layer3b, [0,1,2])
        layer3n=tf.nn.batch_normalization(layer3b, mean3, variance3,offset=None,scale=None,variance_epsilon=1e-5)
        layer3r=tf.nn.relu(layer3n)
#dense layer        
        num_inputs=inputs.shape[0]
        dense_input1=tf.reshape(layer3r,[num_inputs,3*3*20])
        dense_layer1d=tf.add(tf.matmul(dense_input1,self.Wdense1), self.bdense1)
        dense_layer1=tf.nn.dropout(dense_layer1d, 0.3)

        dense_layer2d=tf.add(tf.matmul(dense_layer1,self.Wdense2), self.bdense2)
        dense_layer2=tf.nn.dropout(dense_layer2d, 0.3)

        dense_layer3=tf.add(tf.matmul(dense_layer2,self.Wdense3), self.bdense3)
        # (100,2)

        return dense_layer3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        # print('loss')
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels, logits))/len(logits)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    print('train')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    seeds=[s for s in range(len(train_inputs))]

    index=tf.random.shuffle(seeds)
    train_inputs1=tf.gather(train_inputs,index,axis=0)
    train_labels1=tf.gather(train_labels,index,axis=0)
    train_inputs2=tf.image.random_flip_left_right(train_inputs1, 6).numpy()
    loss1=[]
    for i in range(int(len(train_inputs)/model.batch_size)):
  # Implement backprop:
        with tf.GradientTape() as tape:
            logits=model.call(train_inputs2[i*model.batch_size:(i+1)*model.batch_size])
            losses=model.loss(logits,train_labels1[i*model.batch_size:(i+1)*model.batch_size])
            loss1.append(losses)
      
        gradients = tape.gradient(losses, model.trainable_variables)
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss1

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    # print('test')
    logits = model.call(test_inputs)
    accu = model.accuracy(logits,test_labels)
    return accu


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    train_inputs,train_labels=get_data('/Users/zccc/1470projects/project2/data/train',3,5)
    test_inputs,test_labels=get_data('/Users/zccc/1470projects/project2/data/test',3,5)
    obj=Model()
    for i in range(10):
        train(obj,train_inputs,train_labels)

    accu = test(obj,test_inputs,test_labels)
    print('The accuracy is %f'%accu)

if __name__ == '__main__':
    main()
