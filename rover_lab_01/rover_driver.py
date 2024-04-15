"""
rover_driver.py

# CS-472
# Documentation: The following webpages were referenced for information on various python functions
# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add a header to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
"""

import pyrealsense2.pyrealsense2 as rs
import time
import numpy as np
import cv2
import keras
import utilities.drone_lib as dl

# Path to the trained model weights
MODEL_NAME = "models/rover_model_07_ver01_epoch0049_val_loss0.0006.h5"
#Best so far: "models/rover_model_04_ver01_epoch0003_val_loss9.6779.h5"

# Rover driving command limits
MIN_STEERING, MAX_STEERING = 1100, 1800
MIN_THROTTLE, MAX_THROTTLE = 1100, 1600





# Image processing parameters
white_L, white_H = 220, 255  # White color range
#resizing parameters
resize_W = 320
resize_H = 240

#cropping parameters origin is top left of image so crop_B is the full height of the image (the bottom row visually) and crop_T is the highest part of the image we want to see but the lowest numeric row of pixels we want to see.
crop_W = int(resize_W)
crop_B = resize_H
crop_T = int(resize_H/3)

def get_model(filename):
    """Load and compile the TensorFlow Keras model."""
    model = keras.models.load_model(filename, compile=False)
    model.compile()
    print("Loaded Model")
    return model

def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


def denormalize(steering, throttle):
    """Denormalize steering and throttle values to the rover's command range."""
    steering = invert_min_max_norm(steering, MIN_STEERING, MAX_STEERING)
    throttle = invert_min_max_norm(steering, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def initialize_pipeline():
    """Initialize the RealSense pipeline for video capture."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def get_video_data(pipeline):
    """Capture a video frame, preprocess it, and prepare it for model prediction."""
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None

    #image = np.asanyarray(color_frame.get_data())
    
    #TODO: process your incoming frame so that it is 
    #      in the form required to feed into your CNN.
    color_frame = np.asanyarray(color_frame.get_data())
    #print("Color Frame ", color_frame.shape)

    # resize frame
    color_frame = cv2.resize(color_frame, (resize_W, resize_H))
    #print("Color Frame part 2", color_frame.shape)
    #convert to BW image for easier line detection
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    BW_frame = cv2.inRange(gray_frame, white_L, white_H)

                # Crop Bw image
    BW_frame = BW_frame[crop_T:crop_B, 0:crop_W]
    # Perform cropping (if any)
    # etc...
    image = BW_frame
    #print("BW Frame ",BW_frame.shape )

    return image

def set_rover_data(rover, steering, throttle):
    """Set rover control commands, ensuring they're within the valid range."""
    
    # May uncomment below to force a specific range, if your model is 
    # sometimes outputting weird ranges (probably not needed)
    steering, throttle = check_inputs(int(steering), int(throttle))
    
    rover.channels.overrides = {"1": steering, "3": throttle}
    print(f"Steering: {steering}, Throttle: {throttle}")


def check_inputs(steering, throttle):
    """Check and clamp the steering and throttle inputs to their allowed ranges."""
    steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
    throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def main():

    """Main function to drive the rover based on model predictions."""
   
    # Setup and connect to the rover
    rover = dl.connect_device("/dev/ttyUSB0")

    #HINT:  Get values to the above by querying your own rover...
    throttle_max = rover.parameters['RC3_MAX']
    throttle_min = rover.parameters['RC3_MIN']
    steering_max = rover.parameters['RC1_MAX']
    steering_min = rover.parameters['RC1_MIN']

    print(throttle_max)
    print(throttle_min)
    print(steering_max)
    print(steering_min)

    # Load the trained model
    model = get_model(MODEL_NAME)

    if model is None:
        print("Unable to load CNN model!")
        rover.close()
        print("Terminating program...")
        exit()
        
    while True:
        print("Arm vehicle to start mission.")
        print("(CTRL-C to stop process)")
        while not rover.armed:
            time.sleep(1)
        
        # Initialize video capture
        pipeline = initialize_pipeline()
        
        while rover.armed:
            processed_image = get_video_data(pipeline)
            #sprint("Processed Image size ", processed_image.shape)
            if processed_image is None:
                print("No image from camera.")
                continue

            processed_image = np.expand_dims(processed_image, axis = -1)
            processed_image = np.expand_dims(processed_image, axis = 0)

            # Predict steering and throttle from the processed image
            output = model.predict(processed_image)
            steering, throttle = denormalize(output[0][0], output[0][1])

            # Send commands to the rover
            set_rover_data(rover, int(steering), int(throttle))

        # stop recording
        pipeline.stop()
        time.sleep(1)
        pipeline = None
        rover.close()
        print("Done.")

if __name__ == "__main__":
    main()
