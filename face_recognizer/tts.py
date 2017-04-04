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

name_dict = {'BaekSeongho': '백 성호', 'JangYoonseok': '장 윤석', 'KimDaeseoung': '김 대승', 'KimHwiyoung': '김 휘영',
             'KimJinhyung': '김 진형', 'KimKeeyoung': '김 기영', 'KimSeokwon': '김 석원', 'KimSeongphyo': '김 성표',
             'KimTaehee': '김 태희', 'KimYonbe': '김 연배', 'KoAhra': '고 아라', 'KoMinsam': '고 민삼',
             'LeeHyungyu': '이 현규', 'LeeKwanghee': '이 광희', 'LeeSanghun': '이 상훈', 'LeeYuni': '이 유니',
             'NamKyungpil': '남 경필', 'OhSechang': '오 세창', 'ParkDaeyoung': '박 대영','RohHyungki': '노 형기',
             'SeoByungrak': '서 병락', 'Guest': '손님'}

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

                    label = data

                    msg = "안녕하세요."

                    name = ''

                    if label[:5] == 'Guest':
                        name = 'Guest'
                    else:
                        name = name_dict[label]

                    print name
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

                            print('Play tts')
                            p = subprocess.Popen(['play', '/tmp/welcome.mp3'])
                            p.communicate()
                            print('Play done')


                    else:
                        print("Error Code:" + rescode)

    except:
        print('Error')

if __name__ == "__main__":
    main()
