from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model
import argparse


def training(config_path):
    config = read_config(config_path)
    validation_datasize = config['params']['validation_datasize']
    
    (X_train, y_train),(X_valid, y_valid),(X_test, y_test) = get_data(validation_datasize)

    loss = config['params']['loss_function']
    optimizer = config['params']['optimizer']
    metrics = config['params']['metrics']
    
    model = create_model(loss, optimizer, metrics)
    
    
    EPOCH = config['params']['epoch']
    VALIDATION = (X_valid, y_valid)
    
    histroy = model.fit(X_train, y_train, epochs=EPOCH, validation_data=VALIDATION)

if __name__=='__main__':
    # args = argparse.ArgumentParser()
    # args.add_argument("--config", "-c", default="config.yaml")
    
    # parsed_args = args.parse_arg()
    
    training(config_path="config.yaml")