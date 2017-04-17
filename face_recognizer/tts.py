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

label_list_oguest = [d for d in os.listdir(inputDir + '/oguest') if os.path.isdir(inputDir + '/oguest/' + d) and d != 'Unknown']
label_list_oguest.sort()

label_list.extend(label_list_iguest)
label_list.extend(label_list_oguest)

print(label_list)

name_dict = {'BaekSeongho': '백 성호', 'JangYoonseok': '장 윤석', 'KimDaeseoung': '김 대승', 'KimMina': '김 미나', 'KimHwiyoung': '김 휘영',
             'KimJinhyung': '김 진형', 'KimKeeyoung': '김 기영', 'KimSeokwon': '김 석원', 'KimSeongphyo': '김 성표',
             'KimTaehee': '김 태희', 'KoAhra': '고 아라', 'KoMinsam': '고 민삼',
             'LeeKwanghee': '이 광희', 'LeeSanghun': '이 상훈', 'LeeYuni': '이 유니',
             'OhSechang': '오 세창', 'ParkDaeyoung': '박 대영','RohHyungki': '노 형기',
             'SeoByungrak': '서 병락', 'Guest': '손님'}


sentences_A = ['좋은 아침이에요', '커피 한 잔 하세요', '좋은 하루 되세요', '더 웃으세요', '들어와요', '화이팅!', '멋지시네요',
               '즐거운 하루 되세요', '조용한 물이 깊이 흐릅니다.', '지혜는 고통속에 있습니다.', '힘 찬 하루 되세요', '굿모닝', '반가워요', '어서오세요', '옷이 비싸보이네요', '제 이름은 뭘까요?',
               '어서와요', '요즘 책 좀 보시나요?', '어떤 조언도 길게 하지는 마세요', '배움은 재산이랍니다.', '오 멋져요', '보고싶었어요', '사랑해요', '동안이네요']

sentences_B = ['산뜻한 오후 보내세요', '화이팅 하세요', '운동 좀 하세요', '힘드시죠? 힘내세요', '당신은 자랑스러운 사람입니다',
               '즐거운 하루되세요', '아이리 만세', '조금만 더 힘내세요', '잠오면 세수하세요', '어서와요', '반가워요', '오셨어요?',
               '오늘은 무엇을 배웠나요', '시간과 싸우세요', '오늘은 돌아오지 않는답니다.', '할 수 있다고 믿으면 할 수 있습니다.', '말은 마음의 초상입니다.',
               '긍정적인 마인드']

sentences_C = ['점심 드셨나요', '어서와요', '스마일', '오셨어요?', '멋집니다']

sentences_D = ['곧 즐거운 퇴근이에요', '오늘도 수고 많으셨어요']

sentences_E = ['어서 퇴근하셔야죠']

sentences_F = []

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

    print 'Play tts: ', label

    return path
'''
    if 'Guest' in label:
        path = '/tmp/welcom.mp3'
        name = ' 손님 '
        sentence = ''

        if len(label) > 5:
            num = label[5:]
            name = num + ' 번 손님 '

        if current < 12:
            size = len(sentences_A)
            rv = random.randrange(0, size)
            sentence = sentences_A[rv]
        elif current < 13:
            size = len(sentences_C)
            rv = random.randrange(0, size)
            sentence = sentences_C[rv]
        elif current < 18:
            size = len(sentences_B)
            rv = random.randrange(0, size)
            sentence = sentences_B[rv]
        else:
            size = len(sentences_D)
            rv = random.randrange(0, size)
            sentence = sentences_D[rv]

        print 'Play tts: ' + name

        tts = gTTS(text=name + sentence, lang='ko')
        tts.save(path)

    else:
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

        print 'Play tts: ', name_dict[label]

    
    return path
'''

def main():
    print('Listen redis')

    try:
        for item in pub.listen():
            data = item

            #print('Data received')
            if data is not None:
                data = data.get('data')

                if data > 1L:
                    label = data

                    if label is 'warning':
                        tts = gTTS(text='보안 상 이유로 작동되지 않습니다.', lang='ko')
                        tts.save('/var/tmp/warning.mp3')
                        p = subprocess.Popen(['play', path])
                        p.communicate()

                    else:
                        now = datetime.datetime.now()
                        path = get_voice(label, now.hour)
                        p = subprocess.Popen(['play', path])
                        p.communicate()

                    print('Play done')

    except:
        print('Error')

if __name__ == "__main__":
    main()
