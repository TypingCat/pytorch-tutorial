# Transfer Learning for Computer Vision Tutorial
본 튜토리얼은 전이학습을 활용하여 이미지 분류를 CNN에 학습시킨다. 충분한 크기의 데이터셋이 드물기에 CNN을 처음부터 훈련시키는 일은 일반적이지 않다. 보통은 큰 데이터셋에서 ConvNet을 훈련시키고, 이후 관심작업에 대한 초기화 혹은 특징추출을 한다. 크게 두 가지 전이학습 시나리오가 있다:

- Finetuning the ConvNet: ImageNet 등을 사용하여 사전학습된 네트워크에서 시작한다.
- ConvNet as fixed feature extractor: 사전학습된 네트워크의 마지막 FC만 학습시킨다.
