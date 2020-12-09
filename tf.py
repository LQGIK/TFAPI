import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from sklearn.model_selection import train_test_split
import numpy as np
import json

def getModel(modelStruct):
    '''
    Returns a model given the json structure
    '''

    # Check empty values algorithm
    if not modelStruct:
        return None

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
    print(modelStruct)
    if modelStruct['type'] == "sequential":

        model = keras.Sequential()
        input_shape = tf.ones(modelStruct['input_shape'])


        # Build layers
        for JLayer in modelStruct['layers']:            

            # Check Dense Layer
            if JLayer['type'] == 'dense':
                nodes = int(JLayer['nodes'])
                activation_func = JLayer['activation']
                model.add(layers.Dense(nodes, activation=activation_func))

            elif JLayer['type'] == 'dropout':
                keep_prob = float(JLayer['keep_prob'])
                model.add(layers.Dropout(keep_prob))

        # Set input_shape for weights
        y = model(input_shape)


        # Compile the model with weight optimization, loss algorithms and emphasize accuracy
        model.compile(
            optimizer=modelStruct['optimizer'],
            loss=modelStruct['loss'],
            metrics=modelStruct['metrics']
        )

        return model