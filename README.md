# 이미지 유사도 측정

## 사용기술

- python: 3.9.6
- pip: 23.3.1
- matplotlib: 3.8.2
- numpy: 1.26.1
- opencv_python: 4.8.1.78
- requests: 2.31.0
- flask: 3.0.0

## 초기설정

```
// conda
$ conda create --name "python3.9.6" python="3.9.6"
$ conda info --envs
$ conda activate {name}

$ sh ./setup.sh
```

## 실행(Flask)
```
python app.py
```

#### 유사도 측정 실행

python3 image_similarity.py {개발환경} {업로드파일명}

