from flask import Flask, jsonify, request, render_template
import cv2, numpy as np
import os
import requests
from werkzeug.utils import secure_filename
import json
import imghdr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

app = Flask(__name__)


# Const
app.config['IMAGES_DIR_PATH'] = 'static/src/images/'
app.config['API_HOST'] = 'https://test-bigbro-api.lotteon.com'
app.config['IMAGE_S3_URL'] = 'https://test-contents.lotteon.com/module/screenshot'
app.config['IMAGE_EXT'] = '.webp'
app.config['IMAGE_CONVERT_EXT'] = '.jpeg'


# Request Function
def sendApi(path, method):
    url = app.config['API_HOST'] + path
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
    os.system('curl ' + app.config['IMAGE_S3_URL'] + '/' + filename + ' > ' + app.config['IMAGES_DIR_PATH'] + filename)


# ========================== Home (Page) Start
@app.route('/')
def home():
  return render_template('index.html')
# ========================== Home (Page) End


# ========================== 이미지 유사도 측정 Start
## 이미지 유사도 측정 API
@app.route("/api/similarity/images", methods=['POST'])
def hello():
  if 'uploadFile' not in request.files:
    return 'File is missing', 404

  uploadFile = request.files['uploadFile']
  filename = secure_filename(uploadFile.filename)

  if 'jpeg' not in filename.split('.')[1]:
    return 'jpeg 파일만 업로드할 수 있습니다.', 404

  uploadFile.save(os.path.join(app.config['IMAGES_DIR_PATH'], filename))

  # 업로드 이미지 저장 후 유사도 측정 로직 실행
  moduleImageList = getModuleImageList()
  result = isImageProcess(filename, moduleImageList)

  return jsonify(result)

## 모듈 리스트 호출
def getModuleImageList():
    imageList = list()
    try:
        moduleResult = sendApi('/modules?useYn=Y', 'GET')
        if moduleResult.status_code == 200:
            moduleList = dict(json.loads(moduleResult.text)).get('data')
            for element in moduleList:
                filename = element['dcornNo'] + app.config['IMAGE_EXT']
                convertFileName = element['dcornNo'] + app.config['IMAGE_CONVERT_EXT']
                
                imageList.append(convertFileName)
                if os.path.isfile(app.config['IMAGES_DIR_PATH'] + convertFileName) == False:
                    # 파일 다운 후 확장자 변경 (.webp -> .jpeg)
                    print('New Download: ' + app.config['IMAGES_DIR_PATH'] + filename)
                    isFileDownload(filename)
                    os.rename(app.config['IMAGES_DIR_PATH'] + filename, app.config['IMAGES_DIR_PATH'] + convertFileName)
    except Exception as e:
        print(e)
    return imageList

## 유사도 측정
#### HISTCMP_CORREL: 1에 가까울수록 유사
#### HISTCMP_CHISQR: 0에 가까울수록 유사
#### HISTCMP_INTERSECT: 값이 클수록 유사
#### HISTCMP_BHATTACHARYYA: 0에 가까울수록 유사 (선택)
def imageSimilarity(origImgName, selectImg, modelVal):
    result = {}
    try:
        if imghdr.what(app.config['IMAGES_DIR_PATH'] + selectImg) != None:
            img1 = cv2.imread(app.config['IMAGES_DIR_PATH'] + origImgName)
            img2 = cv2.imread(app.config['IMAGES_DIR_PATH'] + selectImg)
        
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

## 유사도 측정 및 데이터 가공
# 데이터 가공
def isImageProcess(selectImg, imageList):
    imageDictionary = {}
    for image in imageList:
        # 0에 가까울수록 유사
        similarityResult = imageSimilarity(selectImg, image, cv2.HISTCMP_BHATTACHARYYA)

        if not similarityResult:
            print('similarityResult 없음: ' + image)
        else:
            imageDictionary[image] = similarityResult[image]
    return sorted(imageDictionary.items(), key=lambda x:x[1], reverse=False)[:10]
## ========================== 이미지 유사도 측정 End

if __name__ == '__main__':
  app.run('0.0.0.0', port=8000, debug=True)