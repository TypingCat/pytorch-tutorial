# 학습
파이토치 신경망의 학습에는 입력, 목표, 네트워크가 필요하다. 손실 함수와 최적화 기법은 파이토치가 제공한다.

``` py
import torch

# 신경망 구성
network = Network()     # 2장 참조
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01)

# 입력과 목표 준비
input = torch.randn(1, 1, 32, 32)
target = torch.randn(10).view(1, -1)

# 손실 계산
output = network(input)
loss = criterion(output, target)

# 학습
optimizer.zero_grad()   # 변화도 초기화
loss.backward()         # 역전파
optimizer.step()        # 가중치 갱신
```


## 손실 함수
- `torch.nn.MSELoss()`: Mean Squared Error
- `torch.nn.CrossEntropyLoss()`: Cross Entropy


## 최적화 기법
- `torch.optim.SGD(파라미터)`: Stochastic Gradient Descent
- `torch.optim.Adam(파라미터)`: A Method for Stochastic Optimizer
