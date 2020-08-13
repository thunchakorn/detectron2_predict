import os
from utils.broker import RpcClient
# from utils.visualizer import draw_text
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help = 'image path', required=True)
    args = parser.parse_args()

    RPC_HOST = "amqp://user:public@0.0.0.0:5672/"

    # Input file in container
    input_filename = args.image
    print(os.getcwd()+input_filename)
    # Load image in host
    imageshow = cv2.imread(os.getcwd()+input_filename)
    # filename in host
    filename = os.path.splitext(os.path.basename(input_filename))[0]
    # RPC caller
    engine = RpcClient(RPC_HOST)

    # Test Invoice layout
    print("Test calling ctpn")
    messages = {'image_file':input_filename,
                'thresh':0.5}
    results = engine.call('layout_detection',**messages)
    print(results)
    print(len(results))
    # Draw Images
    # image_ctpn = imageshow.copy()
    # for i in results['box']:
    #     points = np.array(i).astype(np.int).reshape(-1,1,2)
    #     cv2.polylines(image_ctpn,[points],True,(0,255,0),3)
    # cv2.imwrite(os.path.join('outputs','{}-ctpn.jpg'.format(filename)),image_ctpn)