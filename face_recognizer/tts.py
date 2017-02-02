#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import urllib2
import redis
import time
import datetime
import subprocess
import random

HOST, PORT = "10.100.1.150", 6379
client_id = "m59KXbFvFqBB5aaTvZea"
client_secret = "M66UpJ4Gr5"

fileDir = os.path.dirname(os.path.realpath(__file__))
baseDir = fileDir + '/../'
inputDir = baseDir + 'face_register/input'

label_list = [d for d in os.listdir(inputDir) if os.path.isdir(inputDir + '/' + d) and d != 'Unknown']
label_list.sort()
print(label_list)

name_list = ['고아라', '서병락', '김대승', '박대영', '김휘영', '노형기', '이현규', '이장형', '김진형', '김기영', '이광희', '남경필', '고민삼', '이상훈', '백성호',  '김태희', '김연배', '장윤석']

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

                if data > 1L:
                    #pub.unsubscribe('tts')
                    index = int(data)
                    print('received ' + str(index))

                    msg = "안녕하세요."
                    name = name_list[index]

                    now = datetime.datetime.now()

                    rv = random.randrange(1, 10)

                    if now.hour < 11:
                        if rv == 1:
                            sentence = '좋은 아침이에요'
                        elif rv == 2:
                            sentence = '커피 한 잔 하세요'
                        elif rv == 3:
                            sentence = '좋은 하루되세요'
                        elif rv == 4:
                            sentence = '날씨가 좋네요'
                        elif rv == 5:
                            sentence = '너무 춥네요'
                        elif rv == 6:
                            sentence = '화이팅'
                        elif rv == 7:
                            sentence = '멋지네요'
                        elif rv == 8:
                            sentence = '즐거운 하루'
                        else:
                            sentence = '힘찬 하루'
                    elif now.hour < 12:
                        sentence = '즐거운 점심'
                    elif now.hour < 17:
                        if rv == 1:
                            sentence = '좋은 오후 보내세요'
                        elif rv == 2:
                            sentence = '커피 한 잔 하세요'
                        elif rv == 3:
                            sentence = '운동 좀 하세요'
                        elif rv == 4:
                            sentence = '날씨가 좋네요'
                        elif rv == 5:
                            sentence = '너무 춥네요'
                        elif rv == 6:
                            sentence = '감기 조심하세요'
                        elif rv == 7:
                            sentence = '산책 어떠세요'
                        elif rv == 8:
                            sentence = '쉬엄쉬엄 하세요'
                        else:
                            sentence = '졸지마세요'
                    else:
                        sentence = '퇴근하세요'

                    msg = msg + name + '님 ' + sentence

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
                            #p = subprocess.Popen(['vlc', '-vvv', '/tmp/welcome.mp3'])
                            print('Play tts')
                            p = subprocess.Popen(['play', '/tmp/welcome.mp3'])
                            p.communicate()
                            print('Play done')
                            #time.sleep(4)
                            #p.terminate()

                    else:
                        print("Error Code:" + rescode)

                    #pub.subscribe('tts')

    except:
        print('Error')

if __name__ == "__main__":
    main()
