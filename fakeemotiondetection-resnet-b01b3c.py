#!/usr/bin/env python
# coding: utf-8

# # Detecting Fake Emotions Using ResNets

# ### importing the required packages

# In[22]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf

from seaborn import *


import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras import layers


# In[23]:


train_dir = "../input/emotion-detection-fer/train"
test_dir = "../input/emotion-detection-fer/test"


# In[24]:


# since the images are 48 * 48 pixels so we can use 48 as side length
side = 48
# the classes in the dataset
classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
# in this dataset we have 7 classes which are [angry, disgusted, fearful, happy, sad, surprised, neutral]
no_classes = 7


# In[25]:


def emotion_dist(path):
    frequencies = {}
    emotions = []
    freq = []
    for emotion in os.listdir(path):
        emotions.append(emotion)
        freq.append(len(os.listdir(path + '/' + emotion)))
        #frequencies[emotion] = len(os.listdir(path + expression))
    plt.bar(emotions,freq, align='center')
    print(freq)
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')
    plt.show()
    
emotion_dist(train_dir)
emotion_dist(test_dir)
    
        


# ### visualize some images

# In[26]:


#visualizing random images from the training set
figure(figsize=(20, 20))
for i in range(10):
    plt.subplot(1,10,i+1)
    rand_class = np.random.randint(0, 6)
    # 110 here is because the folder with the least no. of images has 110 images
    rand_im = np.random.randint(0, 110)
    fname = "/im" + str(rand_im) + ".png"
    image = Image.open(train_dir + '/' + classes[rand_class] + fname).convert("L")
    arr = np.asarray(image)
    plt.title(classes[rand_class])
    plt.axis('off')
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)


# ### perform image augmentation

# In[27]:


# implementing on the fly data augmentation using ImageDataGetnerator
train_generator = ImageDataGenerator(rotation_range= 30,
                              rescale= 1.0/255,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              validation_split = 0.3)

training_data = train_generator.flow_from_directory(train_dir,
                                             target_size=(side,side),
                                             batch_size=16,
                                             color_mode = "grayscale",
                                             class_mode = "categorical",
                                             subset = 'training')

validation_data = train_generator.flow_from_directory(train_dir,
                                             target_size=(side,side),
                                             batch_size=16,
                                             color_mode = "grayscale",
                                             class_mode = "categorical",
                                             subset = 'validation')


# In[28]:


# implementing on the fly data augmentation using ImageDataGetnerator for validation and testing
test_generator = ImageDataGenerator(rescale= 1.0/255)


test_data = test_generator.flow_from_directory(test_dir,
                                             target_size=(side,side),
                                             batch_size=16,
                                             color_mode = "grayscale",
                                             class_mode = "categorical")


# ## The model

# The Model that will be implemented is a ResNet model. The model will consist of multiple identity blocks and convolutional blocks which use the idea of skip connections to handle deep networks gradients. The two blocks mentioned differ in that the identity block input has the same shape as the output of the middle convolutions so we can add them directly, while in the case of convolutional block we do a conv on the input to match the output as described in the figures above each block.
# 
# Note: Model components are reused from an assignment that I had recently solved.

# ### Identity block

# In[29]:



def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block
    X --> Input tensor
    f --> The kernel size of the middle convolution
    filters --> The number of filters that will be used in each of the three CONV layers
    training --> 1 for training, 0 for inference
    initializer --> How to set the initial weights of the network
    
    X as output is the output of this identity block
    """
    
    # Getting the number of filters that will be used in each conv
    F1, F2, F3 = filters
    
    # Saving the values of the inputs in order to use it in the skip connection
    X_shortcut = X
    
    # The convolutions of this block
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X) 

    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    
    # Add the values that we saved to the ones that we have
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = Activation('relu')(X) 

    return X


# ### Convolutional block

# In[30]:



def convolutional_block(X, f, filters, s = 2, training=True, initializer=random_uniform):
    """
    Implementation of the identity block
    X --> Input tensor
    f --> The kernel size of the middle conolution
    filters --> The number of filters that will be used in each of the three CONV layers
    s --> specifiying the stride
    training --> 1 for training, 0 for inference
    initializer --> How to set the initial weights of the network
    
    X as output is the output of this identity block
    """
    
    # Getting the number of filters that will be used in each conv
    F1, F2, F3 = filters
    
    # Saving the values of the inputs in order to use it in the skip connection
    X_shortcut = X

    # The convolutions of this block
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0))(X) 
    X = BatchNormalization(axis = 3)(X, training=training) 
    X = Activation('relu')(X) 

    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X) 
    X = BatchNormalization(axis = 3)(X, training=training) 
    
    # The convolution on the input to be added as a skip connection
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training=training) 

    # Add the values that we saved to the ones that we have
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


# ## ResNet architecture

# ### Why 22?
# 
# The reduction from the conventional ResNet50 to Resnet22 is just for the sole purpose of making training finish sooner. In this project accuracy is not a priority but rather applying concepts.
# 
# ### The flow
# 
# Since this is a ResNet, it uses skip connections which happen in convolutional blocks and identity blocks (These two kinds of blocks were explained earlier). 

# In[31]:



def ResNet22(input_shape, classes):
    """
    input_shape --> shape of the images of the dataset
    classes --> number of classes

    model as output is a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2) 
    X = identity_block(X, 3, [128, 128, 512]) 
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512]) 

    X = AveragePooling2D(pool_size = (2, 2))(X) 
    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X)

    return model


# In[32]:


model = ResNet22(input_shape = (side, side, 1), classes = no_classes)
model.summary()


# In[33]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[34]:


history = model.fit(training_data, epochs=45, validation_data = validation_data)


# ## Model evaluation

# ### Accuracy plot

# In[35]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


# ### Evaluating over testing data

# In[36]:


model.evaluate(test_data)
preds = model.predict(test_data)


# **Note: From the different metrics that we have we can say that this model might, with high confidence, reach better accuracies with deeper architecture. A deeper architecture means more blocks in the resnet. However, as said the concept is the important part here not the accuracy.**

# Plotting the confusion matrix for the classes

# In[37]:


from sklearn import metrics
y = test_data.classes[test_data.index_array]
y_preds = []
for i in preds:
    y_preds.append(np.argmax(i))


# In[38]:


#for i in range(len(y)):
#    print(y[i], y_preds[i])
                   
df= pd.DataFrame(list(zip(y, y_preds)),
               columns =['Orignal', 'predicted']) #Prediction According to calculation

print(df.head(10))

classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

df['Orignal'] = df['Orignal'].replace([0, 1, 2, 3, 4, 5, 6],classes)
df['predicted'] = df['predicted'].replace([0, 1, 2, 3, 4, 5, 6],classes)
print(df.head(10))


# In[39]:


conf_mat = metrics.confusion_matrix(y, y_preds)
conf_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat)
conf_matrix.plot()
plt.show()


# In[40]:


len(y_preds)
    


# In[ ]:




