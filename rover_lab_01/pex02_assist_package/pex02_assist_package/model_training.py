"""
model_training.py

# CS-472
# Documentation: The following webpages were referenced for information on various python functions
# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add a header to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
"""

import data_gen
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

# Configuration parameters
DEVICE = "/GPU:0"  # Device to use for computation. Change to "/GPU:0" if GPU is available
DATA_PATH = '/media/usafa/data/rover_data_processed'  # Path to the processed data
MODEL_NUM = 1 # Model number for naming
TRAINING_VER = 1  # Training version for naming
NUM_EPOCHS = 30  # Number of epochs to train
BATCH_SIZE = 13  # Batch size for training
TRAIN_VAL_SPLIT = 0.8  # Train/validation split ratio


# Define the CNN model structure
def define_model(input_shape=(80, 160, 1)):
    
    # Build you model here...
    model = Sequential([
        Conv2D(16,(3,3),activation = "relu", input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(16,(3,3),activation = "relu"),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(32, activation = "relu"),
        Dropout(0.3),
        Dense(16, activation = "relu"),
        Dropout(0.1),
        Dense(2)

    ])
    
    #opt = SGD(learning_rate = 0.0005, momentum = 0.8)
    model.compile(
        loss = "mse",
        optimizer = "adam",
        metrics=[tf.metrics.mean_absolute_error, tf.metrics.mean_absolute_percentage_error]
    )
    #Do not forget to compile your model here!
    # Compile and select an aoptimizer (probably Adam() and a loss - try mean squared error

    return model

# Train the model with data from a generator, using checkpoints and a specified device
def train_model(amt_data=1.0):
    
    # Load samples (i.e. preprocessed frames for training).
    # Note that we are using sequences consisting of 13 frames.
    samples = data_gen.get_sequence_samples(DATA_PATH, sequence_size=13)
    
    # You may wish to do simple testing using only 
    # a fraction of your training data...
    if amt_data < 1.0:
        # Use only a portion of the entire dataset
        samples, _\
            = data_gen.split_samples(samples, fraction=amt_data)

    # Now, split our samples into training and validation sets
    # Note that train_samples will contain a flat list of sequenced 
    # image file paths.
    train_samples, val_samples = data_gen.split_samples(samples, fraction=TRAIN_VAL_SPLIT)

    train_steps = int(len(train_samples) / BATCH_SIZE)
    val_steps = int(len(val_samples) / BATCH_SIZE)

    # Create data generators that will supply both the training and validation data during training.
    train_gen = data_gen.batch_generator(train_samples, batch_size=BATCH_SIZE)
    val_gen = data_gen.batch_generator(val_samples, batch_size=BATCH_SIZE)
    
    with tf.device(DEVICE):

        # Note that your input shape must match your preprocessed image size
        model = define_model(input_shape=(160, 320, 1))
        model.summary()  # Print a summary of the model architecture
        
        # Path for saving the best model checkpoints
        filePath = "models/rover_model_" + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}" + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        
        # Save only the best (i.e. min validation loss) epochs
        checkpoint_best = ModelCheckpoint(filePath, monitor="val_loss", 
                                          verbose=1, save_best_only=True, 
                                          mode="min")
        
        # Train your model here.
        #print("Made it here 1")
        print(train_gen)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=NUM_EPOCHS,
            verbose=1,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=[checkpoint_best]
        )

        print(history.history.keys())
    return history


# Plot training and validation loss over epochs
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(len(histories),1,1)
        pyplot.title('Training Loss Curves')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train_loss')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='val_loss')

    pyplot.show()

# Run the training process and display training diagnostics
def main():
    
    # Note that you can test your model by using only a portion of the dataset.
    # Set amt_data to whatever proportion of the whole you'd like to train with.
    history = train_model(amt_data=1.0)
    summarize_diagnostics([history])


# Entry point to start the training process
if __name__ == "__main__":
    main()
