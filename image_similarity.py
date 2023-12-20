import os
import cv2 
import pickle
import matplotlib.pyplot as plt

IMAGES_DIR_PATH = 'data/images'
DESCRIPTORS_DIR_PATH = 'data/descriptors'
KEYPOINTS_DIR_PATH = 'data/keypoints'
if not os.path.exists(DESCRIPTORS_DIR_PATH):
    os.makedirs(DESCRIPTORS_DIR_PATH)
if not os.path.exists(KEYPOINTS_DIR_PATH):
    os.makedirs(KEYPOINTS_DIR_PATH)    

# 이미지 크기 조정 기능
def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

def imageResizeTest(image):
    maxD = 1024
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

# 이미지 목록 준비 (키포인트 및 설명자 생성)
imageList = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png"]
imagesBW = []
for imageName in imageList:
    imagePath = IMAGES_DIR_PATH + "/" + str(imageName)
    imagesBW.append(imageResizeTrain(cv2.imread(imagePath,0)))

# opencv 사용
sift = cv2.SIFT_create()
def computeSIFT(image):
    return sift.detectAndCompute(image, None)    

# 다음은 키포인트와 설명자를 생성하는 주요 기능입니다. SIFT를 사용하면 계산하는 데 많은 시간이 걸립니다. 따라서 일단 계산된 값을 저장하는 것이 좋습니다.
keypoints = []
descriptors = []
for i,image in enumerate(imagesBW):
    #print("Starting for image: " + imageList[i])
    keypointTemp, descriptorTemp = computeSIFT(image)
    keypoints.append(keypointTemp)
    descriptors.append(descriptorTemp)
    #print("  Ending for image: " + imageList[i])

# 나중에 사용할 수 있도록 키포인트와 설명자를 저장하세요.
for i,keypoint in enumerate(keypoints):
    deserializedKeypoints = []
    filepath = KEYPOINTS_DIR_PATH + "/" + str(imageList[i].split('.')[0]) + ".txt"
    for point in keypoint:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        deserializedKeypoints.append(temp)
    with open(filepath, 'wb') as fp:
        pickle.dump(deserializedKeypoints, fp)        
for i,descriptor in enumerate(descriptors):
    filepath = DESCRIPTORS_DIR_PATH + "/" + str(imageList[i].split('.')[0]) + ".txt"
    with open(filepath, 'wb') as fp:
        pickle.dump(descriptor, fp)        

# 결과 가져오기 준비
# 저장된 파일에서 키포인트 및 설명자를 가져옴
def fetchKeypointFromFile(i):
    filepath = KEYPOINTS_DIR_PATH + "/" + str(imageList[i].split('.')[0]) + ".txt"
    keypoint = []
    file = open(filepath,'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint        
def fetchDescriptorFromFile(i):
    filepath = DESCRIPTORS_DIR_PATH + "/" + str(imageList[i].split('.')[0]) + ".txt"
    file = open(filepath,'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor

# 결과계산
def calculateResultsFor(i,j):
    keypoint1 = fetchKeypointFromFile(i)
    descriptor1 = fetchDescriptorFromFile(i)
    keypoint2 = fetchKeypointFromFile(j)
    descriptor2 = fetchDescriptorFromFile(j)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(i,j,keypoint1,keypoint2,matches)

    # 결과 데이터
    # print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))   // 좌표값 노출
    print('score ' + str(i) + ' >> ' + str(j) + ': ' + str(score))
    # 비교 이미지 노출
    plt.imshow(plot),plt.show()
def getPlotFor(i,j,keypoint1,keypoint2,matches):
    image1 = imageResizeTest(cv2.imread(IMAGES_DIR_PATH + "/" + imageList[i]))
    image2 = imageResizeTest(cv2.imread(IMAGES_DIR_PATH + "/" + imageList[j]))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)    
# (기본점수 측정항목은 10보다 큰 점수는 매우 좋음을 의미)
def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

# KNN 알고리즘 사용
bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults
def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [255,255,255],
        flags=2
    )
    return matchPlot

# 결과값
print('=============== result')
calculateResultsFor(1, 1)
calculateResultsFor(1, 2)
calculateResultsFor(1, 3)
calculateResultsFor(1, 4)
calculateResultsFor(1, 5)
calculateResultsFor(1, 6)