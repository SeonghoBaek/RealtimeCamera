#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import urllib2
import redis
import time
import subprocess

HOST, PORT = "127.0.0.1", 6379
client_id = "m59KXbFvFqBB5aaTvZea"
client_secret = "M66UpJ4Gr5"

fileDir = os.path.dirname(os.path.realpath(__file__))
baseDir = fileDir + '/../'
inputDir = baseDir + 'face_register/input'

label_list = [d for d in os.listdir(inputDir) if os.path.isdir(inputDir + '/' + d) and d != 'Unknown']
print(label_list)

name_list = ['박대영', '이현규', '김진형', '이광희', '남경필', '고민삼', '이상훈', '백성호', '김태훈', '장윤석', '노형기',
             '김기영', '이장형', '김연배', '서병락']

try:
    rds = redis.StrictRedis(host=HOST,port=PORT,db=0)
    pub = rds.pubsub()
    pub.subscribe('tts')
    redis_ready = True
    print('Redis Ready')

except:
    redis_ready = False
    print('Redis Error')


def main():
    print('Listen redis')

    try:
        for item in pub.listen():
            data = item

            print('Data received')
            if data is not None:
                data = data.get('data')

                if data != 1L:
                    index = int(data)
                    print('received ' + str(index))

                    msg = "안녕하세요."
                    name = name_list[index]

                    msg = msg + name + '님 집에 가'

                    encText = urllib2.quote(msg)
                    data = "speaker=mijin&speed=0&text=" + encText
                    url = "https://openapi.naver.com/v1/voice/tts.bin"
                    request = urllib2.Request(url)

                    request.add_header("X-Naver-Client-Id",client_id)
                    request.add_header("X-Naver-Client-Secret",client_secret)
                    response = urllib2.urlopen(request, data=data.encode('utf-8'))
                    rescode = response.getcode()

                    if rescode == 200:
                        response_body = response.read()

                        with open('/tmp/welcome.mp3', 'wb') as f:
                            print("TTS mp3 save")
                            f.write(response_body)
                            p = subprocess.Popen(['vlc', '-vvv', '/tmp/welcome.mp3'])
                            time.sleep(4)
                            p.terminate()

                    else:
                        print("Error Code:" + rescode)
    except:
        print('Error')

if __name__ == "__main__":
    main()
