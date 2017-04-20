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
from gtts import gTTS

HOST, PORT = "10.100.1.150", 6379
client_id = "m59KXbFvFqBB5aaTvZea"
client_secret = "M66UpJ4Gr5"

fileDir = os.path.dirname(os.path.realpath(__file__))
baseDir = fileDir + '/../'
inputDir = baseDir + 'face_register/input'

label_list = [d for d in os.listdir(inputDir + '/user') if os.path.isdir(inputDir + '/user/' + d) and d != 'Unknown']
label_list.sort()

label_list_iguest = [d for d in os.listdir(inputDir + '/iguest') if os.path.isdir(inputDir + '/iguest/' + d) and d != 'Unknown']
label_list_iguest.sort()

#label_list_oguest = [d for d in os.listdir(inputDir + '/oguest') if os.path.isdir(inputDir + '/oguest/' + d) and d != 'Unknown']
#label_list_oguest.sort()

label_list.extend(label_list_iguest)
#label_list.extend(label_list_oguest)

print(label_list)

name_dict = {'BaekSeongho': '백 성호', 'HyunDaewon': '현 대원', 'JangYoonseok': '장 윤석', 'KimDaeseoung': '김 대승', 'KimMina': '김 미나', 'KimHwiyoung': '김 휘영',
             'KimJinhyung': '김 진형', 'KimKeeyoung': '김 기영', 'KimSeokwon': '김 석원', 'KimSeongphyo': '김 성표',
             'KimTaehee': '김 태희', 'KoAhra': '고 아라', 'KoMinsam': '고 민삼',
             'LeeKwanghee': '이 광희', 'LeeSanghun': '이 상훈', 'LeeYuni': '이 유니',
             'OhSechang': '오 세창', 'ParkDaeyoung': '박 대영','RohHyungki': '노 형기',
             'SeoByungrak': '서 병락', 'Guest': '손님'}

try:
    rds = redis.StrictRedis(host=HOST, port=PORT, db=0)

    pub = rds.pubsub()
    pub.subscribe('tts')
    redis_ready = True
    print('Redis Ready')

except:
    redis_ready = False
    print('Redis Error')


def get_voice(label, current):
    path = './voice/' + label + '/'

    if current < 12:
        path = path + 'A'
        size = len(os.listdir(path))
        rv = random.randrange(0, size)
        path = path + '/' + str(rv) + '_1.mp3'
    elif current < 13:
        path = path + 'C'
        size = len(os.listdir(path))
        rv = random.randrange(0, size)
        path = path + '/' + str(rv) + '_1.mp3'
    elif current < 18:
        path = path + 'B'
        size = len(os.listdir(path))
        rv = random.randrange(0, size)
        path = path + '/' + str(rv) + '_1.mp3'
    else:
        path = path + 'D'
        size = len(os.listdir(path))
        rv = random.randrange(0, size)
        path = path + '/' + str(rv) + '_1.mp3'

    if 'Guest' in label:
        print 'Play tts: ', '손님 ' + label[5:]
    else:
        print 'Play tts: ', name_dict[label]

    return path


def main():
    print('Listen redis')

    try:
        for item in pub.listen():
            try:
                data = item

                if data is not None:
                    data = data.get('data')

                    if data > 1L:
                        label = data

                        if label is 'warning':
                            path = '/var/tmp/warning.mp3'
                            tts = gTTS(text='보안 상 이유로 작동되지 않습니다.', lang='ko')
                            tts.save(path)
                            p = subprocess.Popen(['play', path])
                            p.communicate()

                        else:
                            now = datetime.datetime.now()
                            path = get_voice(label, now.hour)
                            p = subprocess.Popen(['play', path])
                            p.communicate()

                        print('Play done')

            except subprocess.CalledProcessError:
                print('\nSubprocess Error')

    except:
        print('\nRedis Disconnected')

if __name__ == "__main__":
    main()
