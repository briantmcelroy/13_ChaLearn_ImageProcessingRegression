
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def load_train(path):
    
    """
    It loads the train part of dataset from path
    """
    # Init file paths, data generator, and label/target dataframe
    photopath = path + 'final_files/'
    labelpath = path + 'labels.csv'
    datagen = ImageDataGenerator(
        validation_split=0.25, 
        horizontal_flip=True,
        rescale=1/255
    )
    target_labels = pd.read_csv(labelpath)
    
    # Init flow object from dataframe, not from directory, to map images to the age data contained in 'target_labels'
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=target_labels,
        directory=photopath,
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345
    )

    return train_gen_flow


def load_test(path):
    
    """
    It loads the validation/test part of dataset from path
    """
    # Init file paths, data generator, and label/target dataframe
    photopath = path + 'final_files/'
    labelpath = path + 'labels.csv'
    datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255
    )
    target_labels = pd.read_csv(labelpath)
    
    # Init flow object from dataframe, not from directory, to map images to the age data contained in 'target_labels'
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=target_labels,
        directory=photopath,
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345
    )

    return test_gen_flow


def create_model(input_shape):
    
    """
    It defines the model
    """
    # Init ResNet50 architecture with some layers frozen to prevent overtraining
    backbone = ResNet50(
        input_shape=input_shape, 
        weights='imagenet', 
        include_top=False
    )
    backbone.trainable = False
    
    # Create model
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    model.compile(
        optimizer=Adam(learning_rate=0.01), 
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):
    
    # Trying steps_per_epoch and validation_steps as batch size per documentation recommendation
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data) // epochs
    if validation_steps is None:
        validation_steps = len(test_data) // epochs
    
    # Borrowing code for a learning rate scheduler and early stopping for better training control
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    callbacks = [lr_scheduler, early_stopping]
    
    """
    Trains the model given the parameters
    """
    # Assumes train_data and test_data are a tuple, NOT the generator objects to read from the directories.
    # We MUST call the next() function and pass the generator objects to acquire the train_data tuple (outside of this function)
    features_train, target_train = train_data
    features_test, target_test = test_data
    
    # Time to fit. Leaves as verbose=2 for debugging
    model.fit(features_train, 
              target_train,
              validation_data=(features_test, target_test),
              batch_size=batch_size, 
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, 
              callbacks=callbacks
    )

    return model



if __name__ == "__main__":

    # Init
    mainpath = '/content/drive/MyDrive/datasets/faces/'
    input_shape = (150, 150, 3)
    
    # Get training and test tuples
    train_data_gen = load_train(mainpath)
    train_data = next(train_data_gen)

    test_data_gen = load_test(mainpath)
    test_data = next(test_data_gen)

    # Make model
    model = create_model(input_shape)
    
    # Train the model
    model = train_model(
        model, 
        train_data, 
        test_data,
        epochs=20
    )
    
    # Save the model
    model.save("age_prediction_model.h5")


