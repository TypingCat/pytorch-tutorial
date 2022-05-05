# Quickstart
일단 실행해보자!


## Working with data
파이토치는 자체적으로 데이터셋을 제공한다. 이 예제에서는 영상처리 도메인 데이터셋인 TorchVision의 FashionMNIST를 활용한다.

``` py
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

데이터셋은 데이터로더에 장착하여 신경망에 연결한다. 배치 사이즈를 설정할 수 있다.

``` py
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```


## Creating Models
파이토치의 신경망은 `nn.Module`을 상속하는 클래스로 정의한다. 신경망 구조는 다른 장에서 다룬다.

``` py
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

가능하다면 GPU를 사용하고, 아니면 CPU를 사용하는 모델을 생성한다.

``` py
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
```

## Optimizing the Model Parameters
모델을 학습하기 위해 loss function과 optimizer를 생성한다.

``` py
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

모델 학습과정은 다음과 같다.
1. 배치 단위로 들어온 데이터셋으로부터 예측을 수행한다.
2. 역전파를 통해 예측오차를 계산한다.
3. 모델 파라미터를 조정한다.

``` py
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

학습 중 모델의 성능을 확인하는 과정도 필요하다.

``` py
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

위 과정을 수 에포크 동안 반복한다.

``` py
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
```


## Saving & Loading Models
학습된 모델은 간단히 저장할 수 있다.

``` py
torch.save(model.state_dict(), "model.pth")
```

그리고 다시 불러올 수 있다.

``` py
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```

모델을 사용해보자. 평가 데이터셋 첫번째는 다음과 같이 예측한다.

``` py
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```