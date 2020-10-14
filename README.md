# zero_shot_learning

### Introduction
Zero-shot learning refers to applying supervised learning methods to unseen data. That is, the training set and testing set are disjoint. This is an interesting problem, since in the real world, training data is sparse and it's important that models can generalize well to novel data. This repository explores zero-shot learning using convolutional neural networks on the [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/) dataset (AwA2). AwA2 contains 37322 images of 50 different animal classes, each with 85 labeled attributes (e.g. "black", "small", "walks", "smart". The dataset at the above link provides a testing and training split. Each animal class has a length 85 binary attribute vector.

Instead of training a classifier to predict the animal directly, like in canonical image classification, one can predict attributes. These predicted attributes from the network can be used during inference time to find the closest animal class. Some similarity metrics for binary vectors include Hamming distance (i.e. number of inversions), cosine similarity, and Euclidean distance. I found Euclidean distance to work the best (on the sigmoid probabilities - not thresholded binary vectors).

Apart from animal attributes from AwA dataset, This project also uses vector representation of the animal classes as the semantic space. Pretrained word2vec from Google News (Reference: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)  was used to fetch 300-D vectors for each class. Instead of using attributes present in the dataset, these 300-D vectors were used for training the model. The advantage of this approach is it's scalability. The model is not dependent on annotated attribute values of each classes. 

### Model
This repository uses ResNet50 (without pretraining) as the backbone, and adds a fully-connected layer to output a value for each of the attributes (85) or word2vec (300). ResNet by default outputs a 2048 dimensional feature vector in the fully-connected layer.
 I pass the 85 dimensional output for the case when attributes are being used and 300 dimensional vector in the case when word2vec is used through a sigmoid activation to get probabilities for each attribute. Binary cross-entropy loss is then used to train the network, since we are doing multi-label classification (as opposed to multi-class). 

Typically in transfer learning / other supervised learning settings, one would use a pre-trained model (such as ResNet on ImageNet). Here though, using an ImageNet pre-trained model would violate the spirit of zero-shot learning, since ImageNet and AwA2 share some animal classes, and the idea of zero-shot learning is to be able to classify data unseen by the classifier.

### Training and Performance
I trained the attributes model for 30 epochs at a constant learning rate of 0.000025 using Adam optimizer, and got 43.3% accuracy on the testing set (which is entirely disjoint from the training set). I obtained an accuracy of 35.7% for the word2vec case as it was trained for 20 epochs due to limitations of hardware This is not bad compared to some of the earlier literature using AwA2. I suspect that my results would have been better if I had trained for longer, since the loss function  was still decreasing rapidly at the end of training.

### Python3 Libraries
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- KeyedVectors
-PILImage

### Running the Code
1. Make sure all data is downloaded from the above link. This code assumes that the attribute files and JPEG folder are under `data/`, and the training script is in the same root level as `data/`.
2. Install the above Python libraries, if your system does not already have them.
3. Sample execution: `python train.py -n 25 -et 5 -lr 0.000025 -mn 'model.bin' -opt 'optimizer.bin' -o 'predictions.txt' -bs 24`

Reference: https://github.com/dfan/awa2-zero-shot-learning
