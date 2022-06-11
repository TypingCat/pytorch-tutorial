# 신경망
파이토치는 신경망을 손쉽게 구성할 수 있는 클래스(torch.nn)를 제공한다. 개발자가 할 일은 레이어를 적절하게 정의하고 연결하는 것이다. 파이토치 신경망의 기본적인 구조는 다음과 같다.

``` py
import torch
import torch.nn.functional as F

class Network(torch.nn.Module):     # 신경망 모듈 상속
    def __init__(self):
        """레이어 초기화"""
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        """순방향 전파"""
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

> 위 코드는 손글씨 이미지를 숫자 0~9로 분류하는 신경망이다.


## 레이어
- `torch.nn.Linear(입력, 출력)`: 완전연결 레이어
- `torch.nn.Conv2d(입력, 출력, 커널, 스트라이드)`: 컨볼루션 레이어


## 레이어 연산
- `F.relu(레이어)`: 활성화 함수 Rectified Linear Unit
- `F.sigmoid(레이어)`: 활성화 함수 Sigmoid
