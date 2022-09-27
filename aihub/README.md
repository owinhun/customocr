# aihub json파일 converter 
## image & label생성

### 기본적으로 셀은 순서대로 시행을 권장 
# 경로 설정 
<img width="403" alt="image" src="https://user-images.githubusercontent.com/45085563/192590732-76ffc8b8-9886-4c16-891e-5aedd517b8e5.png">
현재 폴더가 없으면 생성함 
----
<img width="380" alt="image" src="https://user-images.githubusercontent.com/45085563/192590809-8af69bfa-66cc-4959-b96a-1832478e1924.png">
`train_dir: 압축 해제한 json파일 경로 `
`img_train_dir: img가 들어있는 파일 경로`

----
<img width="593" alt="image" src="https://user-images.githubusercontent.com/45085563/192591227-58f282db-b2a1-41cf-bc33-f8f81b4ff6c3.png">

여기서 이미지를 읽어서 list로 저장함 

----
<img width="592" alt="image" src="https://user-images.githubusercontent.com/45085563/192591102-ea290e13-6a35-4c94-b3c1-b59bdad36197.png">
> 이미지를 제대로 읽었다면, json에서 crop할 부분을 가지고 와서 image를 crop , 
+alpha는 resize하고 싶은 크기 비율!+ 이미지가 너무 커서 (어차피 우리 모델의 인풋이 매우 작음. resize를 권장함

+ 제대로 되었다면, aihub/img 라는 폴더에 crop된 이미지가 생성될 것
+ label역시 csv로 생성해서 index별 labeling
