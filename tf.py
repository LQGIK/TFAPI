import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import json

'''
inputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
label = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

# First layer
hid1_size = 128
w1 = tf.Variable(tf.random_normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)

# Second layer
hid2_size = 256
w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=0.5)

# Output layer
wo = tf.Variable(tf.random_normal([2, hid2_size], stddev=0.01), name='wo')
bo = tf.Variable(tf.random_normal([2, 1]), name='bo')
yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))
'''

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Add layers to Sequential Model
    model = tf.keras.models.Sequential([

        # Convolutional layer
        tf.keras.layers.Conv2D(
            50, (3,3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Convolutional layer
        tf.keras.layers.Conv2D(
            50, kernel_size=(3, 3), activation="relu", strides=(1,1), padding="same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Max-Pooling Layer, using 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),

        # Convolutional layer
        tf.keras.layers.Conv2D(
            100, kernel_size=(3, 3), activation="relu", strides=(1,1), padding="same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),


        # 2nd Max-Pooling Layer, using 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(.25),

        # Flatten units for input
        tf.keras.layers.Flatten(),


        # Add hidden layers
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        # Add an output layer
        tf.keras.layers.Dense(2, activation="softmax")

    ])

    # Compile the model with weight optimization, loss algorithms and emphasize accuracy
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def genModel(modelStruct):
    '''
    Returns a model given the json structure
    '''

    # Check empty values algorithm
    if not modelStruct:
        return None



    modelStruct = {
        "type":                     "sequential",
        'input_shape':              (1, 2), 
        'optimizer':                'adam',
        'loss':                     'categorical_crossentropy',
        'metrics':  [
                                    'accuracy',
        ],
        "layers":
            [
                {
                    'type':         "dense",
                    'nodes':        1,
                    'activation':    'relu',
                },
                {
                    'type':         'dense',
                    'nodes':        3,
                    'activation':    'relu',
                },
                {
                    'type':         'dropout',
                    'keep_prob':    0.4
                },
                {
                    'type':         "dense",
                    'nodes':        1,
                    'activation':    'softmax',
                }
            ]
    }
    


    # Data Types
    networkTypes = [
        "seq"
    ]
    layerTypes = [
        "dense", 
        "dropout"
    ]
    sequentialParams = {
        'input_shape':  (1, 2) # or (2)
    }
    denseLayerParams = {
        'type' :        'dense',
        'nodes':        1,
        'activation':   'relu',
        
    }
    dropoutLayerParams = {
        'type':         'dropout',
        'keep_prob':    0.4
    }



    # Check model type
    if json.type == "seq":

        model = keras.Sequential()


        # Build layers
        for JLayer in modelStruct.layers:

            # Check Dense Layer
            if JLayer.type == 'dense':
                model.add(layers.Dense(Jlayer.nodes, activation=Jlayer.activation))

            elif JLayer.type == 'dropout':
                model.add(layers.Dropout(JLayer.keep_prob))

        # Set input_shape for weights
        y = model(modelStruct.input_shape)


        # Compile the model with weight optimization, loss algorithms and emphasize accuracy
        model.compile(
            optimizer=modelStruct.optimizer,
            loss=modelStruct.loss,
            metrics=modelStruct.metrics
        )

        return model