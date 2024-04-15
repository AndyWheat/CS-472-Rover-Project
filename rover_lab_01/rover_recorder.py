# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add aheader to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
# for converting to black and white image - https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
# for cropping images - https://learnopencv.com/cropping-an-image-using-opencv/


import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import datetime
import rosbag
import csv
import random
import time

FILENAME = f'/media/usafa/data/rover_data'

#initialize camera settings and turn on camera
def initialize_pipeline(run):
    pipeline = rs.pipeline()
    config = rs.config()
    #record video to bag file
    config.enable_record_to_file(FILENAME + '/'+ str(run)+ '.bag')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

#use to connect to rover in future but not used right now
def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device

#gets color image frame from stream, displays it, and returns the frame number for use in data organization
def get_video_data(pipeline):

    frame = pipeline.wait_for_frames() #get frame
    color_frame = frame.get_color_frame()

    if not color_frame: #if no frame is available signal to restart loop
        return None

    color_image = np.asanyarray(color_frame.get_data()) #get image from frame data
    color_image_fn = color_frame.get_frame_number()
    cv2.imshow('color', color_image) #display image

    return color_image_fn

#in future collect relavent data form the rover, right now return random ints for each field
def get_rover_data(pipeline, rover):
    if (not rover.channels['3'] is None and not rover.channels['1'] is None):
        throttle = int(rover.channels['3'])
        steering = int(rover.channels['1'])
        heading = rover.heading

    return [throttle, steering, heading]

#add data points to the csv file
def append_data(data, index, data_file):
    field_names = ['index', 'throttle', 'steering', 'heading']
    data_dict = {'index': index, 'throttle': data[0], 'steering': data[1], 'heading': data[2]}
    csv.DictWriter(data_file,  fieldnames=field_names).writerow(data_dict)

def main():
    #get starting timestamp for file naming
    run = datetime.datetime.now()

    port = "/dev/ttyUSB0"
    baud = 115200

    print(f"Testing mavlink connection on port {port}, baud rate= {baud}...")

    rover = connect_device(port)

    #create bag file
    bag = rosbag.Bag(FILENAME + '/'+str(run) +'.bag', 'w')
    #create csv and add headers
    header = ['index', 'throttle', 'steering', 'heading']
    with open(FILENAME + '/'+str(run)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    #reopen csv for the remainder of the run
    data_file = open(FILENAME + '/'+str(run)+'.csv', 'a')

    pipeline = initialize_pipeline(run)

    while rover.armed:
        #get index of current frame
        index = get_video_data(pipeline)
        #restart loop if no frame was available
        if index == None:
            continue
        try:
            #get data from rover: throttle, steering, heading
            data = get_rover_data(pipeline, rover)
            #add data to csv with current frame index
            append_data(data, index, data_file)
        except Exception as e:
            #logging.error("Unexpected error while streaming.", exc_info = True)
            break
    

        #logging.info("Stopping recording...")

#exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    time.sleep(10)
#close csv file
    data_file.close()
    print('done')


main()