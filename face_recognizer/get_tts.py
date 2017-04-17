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

client_id = "m59KXbFvFqBB5aaTvZea"
client_secret = "M66UpJ4Gr5"

name_list = ['BaekSeongho', 'JangYoonseok', 'KimDaeseoung', 'KimMina', 'KimHwiyoung',
             'KimJinhyung', 'KimKeeyoung', 'KimSeokwon', 'KimSeongphyo',
             'KimTaehee', 'KoAhra', 'KoMinsam',
             'LeeKwanghee', 'LeeSanghun', 'LeeYuni',
             'OhSechang', 'ParkDaeyoung', 'RohHyungki',
             'SeoByungrak', 'Guest']

name_dict = {'BaekSeongho': '백 성호님', 'JangYoonseok': '장 윤석님', 'KimDaeseoung': '김 대승님', 'KimMina': '김 미나님', 'KimHwiyoung': '김 휘영님',
             'KimJinhyung': '김 진형님', 'KimKeeyoung': '김 기영님', 'KimSeokwon': '김 석원님', 'KimSeongphyo': '김 성표님',
             'KimTaehee': '김 태희님', 'KimYonbe': '김 연배님', 'KoAhra': '고 아라님', 'KoMinsam': '고 민삼님',
             'LeeHyungyu': '이 현규님', 'LeeKwanghee': '이 광희님', 'LeeSanghun': '이 상훈님', 'LeeYuni': '이 유니님',
             'NamKyungpil': '남 경필님', 'OhSechang': '오 세창님', 'ParkDaeyoung': '박 대영님','RohHyungki': '노 형기님',
             'SeoByungrak': '서 병락님', 'Guest': '손님'}

sentences_A = ['좋은 아침이에요', '커피 한 잔 하세요', '좋은 하루 되세요', '더 웃으세요', '들어와요', '화이팅!', '멋지시네요',
               '즐거운 하루 되세요', '조용한 물이 깊이 흐릅니다.', '지혜는 고통속에 있습니다.', '힘 찬 하루 되세요', '굿모닝', '반가워요', '어서오세요', '옷이 비싸보이네요', '제 이름은 뭘까요?',
               '어서와요', '요즘 책 좀 보시나요?', '어떤 조언도 길게 하지는 마세요', '배움은 재산이랍니다.', '오 멋져요', '보고싶었어요', '사랑해요', '동안이네요']

sentences_B = ['산뜻한 오후 보내세요', '화이팅 하세요', '운동 좀 하세요', '힘드시죠? 힘내세요', '당신은 자랑스러운 사람입니다',
               '즐거운 하루되세요', '아이리 만세', '조금만 더 힘내세요', '잠오면 세수하세요', '어서와요', '반가워요', '오셨어요?',
               '오늘은 무엇을 배웠나요', '시간과 싸우세요', '오늘은 돌아오지 않는답니다.', '할 수 있다고 믿으면 할 수 있습니다.', '말은 마음의 초상입니다.',
               '긍정적인 마인드']

sentences_C = ['점심 드셨나요', '어서와요', '스마일', '오셨어요?', '멋집니다']

sentences_D = ['곧 즐거운 퇴근이에요']

sentences_E = ['어서 퇴근하셔야죠']

sentences_F = []


def save_tts(sentence, path):
    tts = gTTS(text=sentence, lang='ko')
    tts.save(path + '_1.mp3')

    """
    encText = urllib2.quote(sentence)
    data = "speaker=mijin&speed=0&text=" + encText
    url = "https://openapi.naver.com/v1/voice/tts.bin"
    request = urllib2.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib2.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read()

        with open(path + '_2.mp3', 'wb') as f:
            f.write(response_body)
    """

    #time.sleep(1)

def main():
    folder = './voice/'

    if not os.path.exists(folder):
        os.mkdir(folder)

    for i in range(100):
        if not os.path.exists(folder + 'Guest' + str(i)):
            os.mkdir(folder + 'Guest' + str(i))
            name = 'Guest' + str(i)

            os.mkdir(folder + name + '/A')
            os.mkdir(folder + name + '/B')
            os.mkdir(folder + name + '/C')
            os.mkdir(folder + name + '/D')
            os.mkdir(folder + name + '/E')

            sentence = str(i) + ' 번 손님 '
            j = 0

            for s in sentences_A:
                s_a = sentence + ' ' + s
                path = folder + name + '/A/' + str(j)
                save_tts(s_a, path)
                j += 1

            j = 0

            for s in sentences_B:
                s_a = sentence + ' ' + s
                path = folder + name + '/B/' + str(j)
                save_tts(s_a, path)
                j += 1

            j = 0

            for s in sentences_C:
                s_a = sentence + ' ' + s
                path = folder + name + '/C/' + str(j)
                save_tts(s_a, path)
                j += 1

            j = 0

            for s in sentences_D:
                s_a = sentence + ' ' + s
                path = folder + name + '/D/' + str(j)
                save_tts(s_a, path)
                j += 1

            j = 0

            for s in sentences_E:
                s_a = sentence + ' ' + s
                path = folder + name + '/E/' + str(j)
                save_tts(s_a, path)
                j += 1

    for name in name_list:
        if not os.path.exists(folder + name):
            os.mkdir(folder + name)
            os.mkdir(folder + name + '/A')
            os.mkdir(folder + name + '/B')
            os.mkdir(folder + name + '/C')
            os.mkdir(folder + name + '/D')
            os.mkdir(folder + name + '/E')

            sentence = name_dict[name]
            i = 0

            for s in sentences_A:
                s_a = sentence + ' ' + s
                path = folder + name + '/A/' + str(i)
                save_tts(s_a, path)
                i += 1

            i = 0

            for s in sentences_B:
                s_a = sentence + ' ' + s
                path = folder + name + '/B/' + str(i)
                save_tts(s_a, path)
                i += 1

            i = 0

            for s in sentences_C:
                s_a = sentence + ' ' + s
                path = folder + name + '/C/' + str(i)
                save_tts(s_a, path)
                i += 1

            i = 0

            for s in sentences_D:
                s_a = sentence + ' ' + s
                path = folder + name + '/D/' + str(i)
                save_tts(s_a, path)
                i += 1

            i = 0

            for s in sentences_E:
                s_a = sentence + ' ' + s
                path = folder + name + '/E/' + str(i)
                save_tts(s_a, path)
                i += 1


if __name__ == "__main__":
    main()
