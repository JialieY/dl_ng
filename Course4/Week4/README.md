## Art generation with Neural Style Transfer

Activation Matrix to measure content loss
Gram Matrix to measure style loss, coorelation between activation of different filters

VGG 19 
"imagenet-vgg-verydeep-19.mat"
http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

VGG 16
"imagenet-vgg-verydeep-16.mat"
http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat


## Face Recognition

Face Verification - "is this the claimed person?"

Face Recognition - "who is this person?"

Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem.

The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.

The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.
