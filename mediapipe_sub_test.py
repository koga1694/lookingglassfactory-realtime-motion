import zmq
import time
import json
import cv2 as cv

context = zmq.Context()
sockets = context.socket(zmq.SUB)

sockets.connect('tcp://127.0.0.1:10100')
sockets.setsockopt_unicode(zmq.SUBSCRIBE, '') # 모든 메세지를 받는다.


while True:
    dic = sockets.recv_json()
    print(dic)
    print(type(dic))
    if cv.waitKey(5) & 0xFF == 27:
        break
    
