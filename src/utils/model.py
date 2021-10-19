import tensorflow as tf

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