import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from nst_utils import *
import numpy as np
import tensorflow as tf


# Content cost
def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    J_content = 1/(4*n_H* n_W* n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C,a_G)))
    return J_content

# Style cost
def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C =  a_G.get_shape().as_list()
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S,perm=[0,3,2,1]),[n_C,-1])
    a_G = tf.reshape(tf.transpose(a_G,perm=[0,3,2,1]),[n_C,-1])
    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    # Computing the loss (≈1 line)
    J_style_layer = 1/(2*n_H* n_W* n_C)**2*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        print('A_G in style cost', a_G)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha*J_content + beta*J_style
    return J

# Reset the graph
tf.reset_default_graph()
# Start interactive session
sess = tf.InteractiveSession()

# Import the content image and the style image
height = 300
width = 400
content_image_path = 'tubingen.jpg'
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))

style_image_path = 'starry_night.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))

content_image = np.asarray(content_image, dtype='float32')
content_image = reshape_and_normalize_image(content_image)
style_image = np.asarray(style_image, dtype='float32')
style_image = reshape_and_normalize_image(style_image)

# load model
generated_image = generate_noise_image(content_image)
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

# Compute the content cost
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

# Compute the style cost
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, alpha = 10, beta = 40)
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations = 400):
    sess.run(tf.global_variables_initializer())
    generated_image=sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)
        # Print every 20 iteration.
        if i % 20 == 0:
            generated_image = sess.run(model['input'])
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    save_image('generated_image.jpg', generated_image)
    return generated_image

model_nn(sess, generated_image)
