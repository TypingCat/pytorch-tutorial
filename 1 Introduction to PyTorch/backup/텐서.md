# 텐서
파이토치는 텐서(Tensor)라는 자료형을 사용한다. 텐서는 임의로 차원을 설정할 수 있어 사실상 모든 수치화된 데이터를 표현할 수 있다. 파이토치의 텐서는 일반적으로 [배치, 채널, 데이터]의 형태를 갖는다.


## 생성
``` py
t = torch.empty(5, 3)       # 선언
t = torch.rand(5, 3)        # 무작위 초기화

t = torch.zeros(5, 3)       # 영행렬 초기화
t = torch.zeros_like(x)     # 텐서 x와 동일한 크기의 영행렬 초기화
t = torch.ones(5, 3)        # 일행렬 초기화
t = torch.ones_like(x)      # 텐서 x와 동일한 크기의 일행렬 초기화
```


## 타입 변환
``` py
tensor = torch.tensor(list)        	# List --> Tensor
array = numpy.array(list)          	# List --> Array
list = tensor.tolist()              # Tensor --> List
array = tensor.numpy()             	# Tensor --> Array
list = array.tolist()              	# Array --> List
tensor = torch.from_numpy(array)    # Array --> Tensor

tensor = tensor.short()		# int16
tensor = tensor.int()		# int32
tensor = tensor.long()		# int64
tensor = tensor.half()		# float16
tensor = tensor.float()		# float32
tensor = tensor.double()	# float64
```


## 모양 변형
``` py
print(x.shape)              # 모양 확인

x.view(3, -1)               # 모양을 [3, :]로 변경
x.squeeze()                 # 크기가 1인 차원들을 제거
x.unsqueeze(d)              # d번째에 크기가 1인 차원을 추가

z = torch.cat([x, y], dim=d)	# 텐서 x와 y를 d차원에서 연결
```


## 텐서 연산
``` py
z = torch.add(x, y)         # 덧셈
z = torch.matmul(x, y)      # 행렬곱
z = torch.mul(x, y)         # 원소곱
z = torch.sum(x)            # 합
z = torch.mean(x)           # 평균
```
