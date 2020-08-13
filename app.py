import os
import cv2
from engine import INV_layout
import json
import sys
import torch

import amqpstorm
from amqpstorm import Message


#RPC_HOST = amqp://user:public@ocr-mq:5672/
#RPC_HOST = amqp://user:public@localhost:5672/
#RPC_KEY = ztrus_ctpn

# Environment varaible 
RPC_HOST=os.environ.get('RPC_HOST')
RPC_KEY=os.environ.get('RPC_KEY')
is_cuda = torch.cuda.is_available()

print("MQ Host: {}".format(RPC_HOST))
print("MQ Key: {}".format(RPC_KEY))

# Load model
text_detecion_engine = INV_layout(cuda = is_cuda)

# process an image to get ctpn box
def process(image_file, thresh = 0.5):
    img = cv2.imread(image_file)
    print(img.shape)
    results = text_detecion_engine.predict(img=img, thresh = thresh)
    results = json.loads(json.dumps(results))
    return results


def fn_on_request(fn_process):
    def on_request(message):
        if isinstance(message.body, bytes):
            args = message.body.decode("utf-8")
        else:
            args = message.body

        # convert json format to nested format    
        args = json.loads(args)

        # call fn_process with args
        response = json.dumps(fn_process(**args),ensure_ascii=False)

        properties = {
            'correlation_id': message.correlation_id,
            'content_type': 'application/json'
        }
        # Set message
        response = Message.create(message.channel, response, properties)
        response.publish(message.reply_to)
        
        message.ack()

    return on_request

if __name__ == '__main__':
    # Create Connection
    CONNECTION = amqpstorm.UriConnection(RPC_HOST)  
    CHANNEL = CONNECTION.channel()

    CHANNEL.queue.declare(queue=RPC_KEY)
    CHANNEL.basic.qos(prefetch_count=1)
    CHANNEL.basic.consume(fn_on_request(process), queue=RPC_KEY)

    print(" [x] Awaiting RPC requests")
    CHANNEL.start_consuming()
    sys.exit(0)