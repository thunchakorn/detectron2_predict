import os
from utils.broker import RpcClient
from utils.visualizer import visualising
# from utils.visualizer import draw_text
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help = 'image path e.g. ./images/123.jpg')
    args = parser.parse_args()

    RPC_HOST = "amqp://user:public@0.0.0.0:5672/"

    # Input file in container
    # examp
    input_filename = args.image
    print(os.path.join(os.getcwd(), input_filename))
    # Load image in host
    imageshow = cv2.imread(os.path.join(os.getcwd(), input_filename))
    # filename in host
    filename = os.path.splitext(os.path.basename(input_filename))[0]
    # RPC caller
    engine = RpcClient(RPC_HOST)

    # Test Invoice layout
    print("Test calling layout detection")
    messages = {'image_file':input_filename,
                'thresh':0.5}
    results = engine.call('layout_detection',**messages)

    # Draw Images
    print('visualising')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    save_name = os.path.join('outputs','{}-layout-detected.jpg'.format(filename))
    visualising(imageshow, results, save_name)


