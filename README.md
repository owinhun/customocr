## customocr

### 경로 설정 방법: ```customocr```의 하위 경로에 data 파일 위치
.
customocr <br>
├── font <br>
├── hallymocr <br>
├── ocr.ipynb <br>
├── ... <br>
├── open ==> dacon 파일 <br>
├── word_h ==> [구글 드라이브 파일](https://drive.google.com/file/d/1b_ae-2qFmUOo6-QTOapZ4TbPr8iTYYOp/view?usp=sharing) <br>
├── word_v <br>
├── ko.txt <br>
└── train.txt <br>

### valid_data 조절 방법(학습 시간 단축)

1. OCR 모델 생성 전 validation data 생성 코드 추가(```'train/'``` 으로 가정)
```Python
import pandas as pd

###### train.csv를 이용해 개수 조절
train_csv_path = 'open/train.csv'
train_csv = pd.read_csv(train_csv_path)
train_csv[:10].to_csv('open/train.txt', sep='\t', header=False, index=False)


###### lmdb 파일 생성, 운영체제에 맞게 주석 해제
# # if window
# !python ./hallymocr/create_lmdb_dataset.py --inputPath ./open/ --gtFile ./open/train.txt --outputPath ./result/train --file_size <전체 데이터 크기(GB)>
# # if linux
# !python3 ./hallymocr/create_lmdb_dataset.py --inputPath ./open/ --gtFile ./open/train.txt --outputPath ./result/train --file_size <전체 데이터 크기(GB)>
```

2. ```opt['valid_data'] ```수정
```Python
opt = {
    'exp_name': None,
    'train_data': './result/',
    'valid_data': './result/train',
    ...
}

```
----
### - <a href="./aihub/README.md">json_file 컨버터</a>
