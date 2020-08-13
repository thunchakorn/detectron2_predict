# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.append(os.getcwd())

import amqpstorm
from amqpstorm import Message
import uuid
import json

class RpcClient(object):
    def __init__(self, host):
        """
        :param host: RabbitMQ Server e.g. localhost
        :return:
        """
        self.host = host
        self.channel = None
        self.response = None
        self.connection = None
        self.callback_queue = None
        self.correlation_id = None
        self.open()

    def open(self):
        self.connection = amqpstorm.UriConnection(self.host)  

        self.channel = self.connection.channel()

        result = self.channel.queue.declare(exclusive=True)
        self.callback_queue = result['queue']

        self.channel.basic.consume(self._on_response, no_ack=True,
                                   queue=self.callback_queue)

    def close(self):
        self.channel.stop_consuming()
        self.channel.close()
        self.connection.close()

    def call(self, routing_key,**kargs):
        self.response = None
        # create message
        message = json.dumps(kargs,ensure_ascii=False)
        message = Message.create(self.channel, body=message)
        message.reply_to = self.callback_queue
        self.correlation_id = message.correlation_id

        # publish to rabbit-mq
        message.publish(routing_key=routing_key)

        # waiting
        while not self.response:
            self.channel.process_data_events()

        return json.loads(self.response)

    def _on_response(self, message):
        if self.correlation_id != message.correlation_id:
            return
        self.response = message.body