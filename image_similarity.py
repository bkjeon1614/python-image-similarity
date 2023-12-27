# -*- coding: utf-8 -*-
import cv2, numpy as np
import matplotlib.pylab as plt
import os
import sys
import requests
import json
import imghdr

sys.stdout = open('stdout.txt', 'w')

# Const
IMAGES_DIR_PATH = 'data/images/'
API_HOST = 'https://test-bigbro-api.lotteon.com'
IMAGE_S3_URL = 'https://test-contents.lotteon.com/module/screenshot'
IMAGE_EXT = '.webp'
IMAGE_CONVERT_EXT = '.jpeg'

# Request Function
def sendApi(path, method):
    url = API_HOST + path
    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            body = {}
            response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t"))
    except Exception as ex:
        print(ex)            
    return response

# Image URL Download
def isFileDownload(filename):
    os.system('curl ' + IMAGE_S3_URL + '/' + filename + ' > ' + IMAGES_DIR_PATH + filename)

# 모듈 리스트 호출
def getModuleImageList():
    imageList = list()
    try:
        moduleResult = sendApi('/modules?useYn=Y', 'GET')
        if moduleResult.status_code == 200:
            moduleList = dict(json.loads(moduleResult.text)).get('data')
            for element in moduleList:
                filename = element['dcornNo'] + IMAGE_EXT
                convertFileName = element['dcornNo'] + IMAGE_CONVERT_EXT
                
                imageList.append(convertFileName)
                if os.path.isfile(IMAGES_DIR_PATH + convertFileName) == False:
                    # 파일 다운 후 확장자 변경 (.webp -> .jpeg)
                    print('New Download: ' + IMAGES_DIR_PATH + filename)
                    isFileDownload(filename)
                    os.rename(IMAGES_DIR_PATH + filename, IMAGES_DIR_PATH + convertFileName)
    except Exception as e:
        print(e)
    return imageList

# 로컬에 모듈 이미지 리스트 저장
# def setModuleImage():
#     for i in os.listdir(IMAGES_DIR_PATH):
#         fullPath = os.path.join(IMAGES_DIR_PATH, i)
#         if os.path.isfile(fullPath):
#             imageList.append(fullPath.split('/')[2])


def imageSimilarity(origImgName, selectImg, modelVal):
    result = {}
    try:
        if imghdr.what(IMAGES_DIR_PATH + selectImg) != None:
            img1 = cv2.imread(IMAGES_DIR_PATH + origImgName)
            img2 = cv2.imread(IMAGES_DIR_PATH + selectImg)
        
            # ----------img resize---------------------
            img2 = cv2.resize(img2, dsize = (197, 256))

            cv2.imshow('query', img1)
            imgs = [img1, img2]
            hists = []
            for i, img in enumerate(imgs) :
                plt.subplot(1, len(imgs), i+1)
                plt.title('img%d'% (i+1))
                plt.axis('off') 
                plt.imshow(img[:,:,::-1])
                #---① 각 이미지를 HSV로 변환
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                #---② H,S 채널에 대한 히스토그램 계산
                hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
                #---③ 0~1로 정규화
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                hists.append(hist)

            query = hists[0]
            for i, (hist, img) in enumerate(zip(hists, imgs)):
                #---④ 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
                ret = cv2.compareHist(query, hist, modelVal)
                if modelVal == cv2.HISTCMP_INTERSECT: # 교차 분석인 경우 
                    result[selectImg] = ret/np.sum(query)         #비교대상으로 나누어 1로 정규화
                result[selectImg] = round(ret, 5)
    except Exception as e:
        print(e)
    return result

# 데이터 가공
def isImageProcess(selectImg, imageList):
    result = {}
    imageDictionary = {}
    for image in imageList:
        # 0에 가까울수록 유사
        similarityResult = imageSimilarity(selectImg + IMAGE_CONVERT_EXT, image, cv2.HISTCMP_BHATTACHARYYA)

        if not similarityResult:
            print('similarityResult 없음: ' + image)
        else:
            imageDictionary[image] = similarityResult[image]
    imageDictionarySorted = sorted(imageDictionary.items(), key=lambda x:x[1], reverse=False)[:10]
    result = dict(imageDictionarySorted)
    return result

# 결과값 리턴
# HISTCMP_CORREL: 1에 가까울수록 유사
# HISTCMP_CHISQR: 0에 가까울수록 유사
# HISTCMP_INTERSECT: 값이 클수록 유사
# HISTCMP_BHATTACHARYYA: 0에 가까울수록 유사 (선택)
if len(sys.argv) > 2:
    if sys.argv[1] == 'dev' or sys.argv[1] == 'prod':
        indexImageName = sys.argv[2]
        moduleImageList = getModuleImageList()
    else:
        print('올바른 개발환경을 입력하여 주시길 바랍니다. (dev or prod)')

    with open(indexImageName + '.json', 'w') as json_file:
        json.dump(isImageProcess(indexImageName, moduleImageList), json_file)    
else:
    print('개발환경(dev or prod) 또는 모듈번호를 매개변수로 입력하여 주시길 바랍니다. (Ex: python3 image_similarity_new.py dev M001558)')

sys.stdout.close()