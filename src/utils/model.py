import tensorflow as tf
import time as time

import os

def create_model(loss, optimizer, metrics):
    
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
          tf.keras.layers.Dense(300, activation="relu", name="hd1"),
          tf.keras.layers.Dense(100, activation="relu", name="hd2"),
          tf.keras.layers.Dense(10, activation="softmax", name="op")
    ]
    
    model_clf = tf.keras.models.Sequential(LAYERS)
    
    model_clf.summary()
    
    LOSS = loss
    OPTIMIZER = optimizer
    METRICS = [metrics]
    
    model_clf.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    
    return model_clf #<< untrained model

def get_uniq_filename(filename):
    uniq_filename = time.strftime(f"%Y-%m-%dT%H:%S_{filename}")
    return uniq_filename

def save_model(model, model_name,model_dir):
    uniq_filename = get_uniq_filename(model_name)
    path_to_model = os.path.join(model_dir,uniq_filename)
    model.save(path_to_model)
