# 한국 SW중심대학 공동 AI 경진대회 <본선>
## 최종코드 제출
## Team: 한림대학교
## private score: 5위
* 안녕하세요, 한림대학교 팀입니다. 저희는 한림대학교 학부생 동기 5인으로 구성되어 있으며 대회에 참가하면서 좋은 경험을 했습니다. 감사합니다. 아래는 score 재현 방법입니다.
---

# 실행 방법
## 외부 사용 데이터
## **아래 데이터셋을 `TIW` 으로 칭합니다.**
[AIHUB: 한국어 글자체 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=81) </br>

## **아래 데이터셋을 `HUB` 으로 칭합니다.**
[AIHUB: 야외 실제 촬영 한글 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=105) </br>

## **데이콘이 제공한 데이터셋을 `TRAIN`과 `TEST`로 칭합니다.**


### 파일 경로 설정 방법: ./customocr
├── <b>hallymocr: *학습에 필요한 모듈들*</b></br>
│   ├── modules</br>
│   │   └── \_\_pycache__</br>
│   └── \_\_pycache__</br>
├── images: *학습에 사용되는 이미지 파일*</br>
│   ├── crop_tiw : *크롭된 `HUB` 이미지*</br>
│   ├── crop_train : *크롭된 `TIW` 이미지*</br>
│   ├── hub_train : *`HUB`이미지*</br>
│   ├── test : *`TEST`이미지*</br>
│   ├── tiw : *`TIW` 원본이미지*</br>
│   └── train : *`TRAIN` 원본이미지*</br>
├── labels: *라벨 및 이미지 정보가 담긴 json파일*</br>
│   ├── hub_train : *`HUB` json파일*</br>
│   └── tiw : *`TIW` json파일*</br>
├── lmdb_gt: *lmdb생성을 위한 txt파일*</br>
├── result: *생성된 lmdb파일*</br>
│   ├── htrain: *`HUB` lmdb*</br>
│   ├── tiw: *`TIW` lmdb*</br>
│   ├── train: *`TRAIN[:-1000]`*</br>
│   └── valid: *`TRAIN[-1000:]`*</br>
├── saved_models: *모델 및 로그 저장*</br>
│   └── TPS-ResNet-BiLSTM-Attn-Seed1111</br>
├── <b>ocr.ipynb: *핵심 실행 파일(main)*</b></br>
├── README.md</br>
├── requirements.yaml</br>
└── train_edit.csv: 수작업으로 정제된 `TRAIN` csv파일</br>


## 1. git pull하기
```linux
git pull https://github.com/mhseo10/customocr
```

## 2. AI_HUB에서 데이터 다운로드 받기 </br>
[한국어 글자체 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=81) </br>
[야외 실제 촬영 한글 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=105)

## 3. 이미지 및 json파일 저장 (★★)
### 한국어 글자체 이미지 (`TIW`)
- ★ 중요 ★4개의 카테고리중 `Text in the wild` 데이터셋만 사용
- 이미지 파일:  `./images/tiw`에 저장하며, 이때 `Text in the wild`데이터 셋 안에 4개의 폴더가 있는데, 이를 각 폴더로 두지 않고 각 폴더안에 있는 image파일들을 `./tiw`에 한 곳으로 저장해야합니다.
- json 파일: `TIW` 데이터셋은 하나의 json파일로 라벨링이 이루어져 있습니다.
    - ex) 100만장의 image, 1개의 json파일
    - 이때 json파일은 `textinthewild_data_info.json`입니다. 
    - `./labels/tiw`에 저장합니다.
### 야외 실제 촬영 한글 이미지 (`HUB`)
- `HUB`셋은 Train셋과 Validation셋이 존재하나 Train셋만 사용합니다.
- 이미지 파일: `./images/hub_train`에 저장하며 이때 `TIW`와 다르게 다운된 폴더 구조 째로 넣어주시면 됩니다.
- json 파일: KOREAN 데이터셋은 하나의 image에 하나의 json파일이 각각 매치되어있습니다.
    - ex) 100만장의 image, 100만개의 json파일
    - [원천]과 [라벨]로 이름이 지정되어있습니다. [원천]의 이름에 맞게 [라벨]에 json파일이 매치되어있습니다.
    - `./labels/hub_train`에 저장합니다.
## 4. `ocr.ipynb` 노트북 실행