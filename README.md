# PyTorch DCGAN

## THIS IS ENGLISH AND PYTORCH VERSION OF [THIS PORJECT](https://github.com/Won-Been/ai06project4).

## 1. 이미지 전처리
- 캐글에서 드레스 다운 받은 이미지의 사이즈를 (160, 120, 3)으로 줄인 후 추후 사용하기 편하도록 npz파일로 압축 저장한다.
- Download dress image from Kaggle and resize the images to (160, 120).
- Save resized images to .npz file.

- gdown으로 압축 저장한 이미지를 다운받은 후 -1~1사이의 값으로 정규화 한다.
- Download

## 2. Generator, Discriminator 모델 구축

  <img width="565" alt="스크린샷 2022-03-13 오후 4 38 35" src="https://user-images.githubusercontent.com/86759423/158050146-c4c70256-5fe8-4263-b17e-6e643aa551a8.png">
  <img width="582" alt="스크린샷 2022-03-13 오후 4 38 59" src="https://user-images.githubusercontent.com/86759423/158050152-4d695891-924b-4a3c-a4a2-e16038c83db3.png">

## 3. 학습 결과 이미지

  <img width="224" alt="스크린샷 2022-03-13 오후 4 39 58" src="https://user-images.githubusercontent.com/86759423/158050177-91a581ec-bb54-486c-ac3c-02180bff58f2.png">
