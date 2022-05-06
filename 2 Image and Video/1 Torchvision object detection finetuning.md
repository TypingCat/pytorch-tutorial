# Torchvision object detection finetuning
> 튜토리얼 코드 유지보수가 되지 않아 실행은 되지 않는다.

사전학습된 모델의 미세조정을 해 보자. `Penn-Fudan Database for Pedestrian Detection and Segmentation`의 Mask R-CNN을 사용한다. 여기에는 170개 이미지와 345개 보행자 인스턴스가 포함되어 있다. 커스텀 데이터셋으로 인스턴트 분류 모델을 학습시킬 것이다.

``` bash
$ pip install --user pycocotools
```


## Defining the Dataset
참조 스크립트는 커스텀 데이터셋을 추가하기 쉽다. 데이터셋은 `torch.utils.data.Dataset` 클래스를 상속해야 하며, 함수 `__len__`/`__getitem__`을 구현해야 한다. 특히 `__getitem__`은 이하를 반환해야 한다.

- image: 크기 (H, W)인 PIL(Python Imaging Library) 이미지
- target: 이하의 필드를 갖는 딕셔너리
    - boxes: 바운딩 박스의 모서리 좌표 (x0, y0, x1, y1), FloatTensor[N, 4]
    - labels: 각 박스의 레이블, Int64Ternsor[N]
    - image_id: 이미지 식별자, Int64Tensor[1]
    - area: 바운딩 박스의 면적, Tensor[N]
    - iscrowd: 복수의 객체가 포함되었는지 여부, 평가 시 제외, UInt8Tensor[N]
    - (optionally) masks: 각 객체에 대한 분류 마스크, UInt8Tensor[N, H, W]
    - (optionally) keypoints: 각 객체에 대한 K개 키포인트 (x, y, visibility), FloatTensor[N, K, 3]

- 배경 레이블은 0으로 약속되어 있다.
- 종횡비가 비슷한 이미지들로 배치를 구성하려면 함수 `get_height_and_width`를 추가해야 한다.

### Writing a custom dataset for PennFudan
PennFudan 파일을 [다운로드](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)하자. 그리고 여기에 맞는 데이터셋을 작성한다.

``` py
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        ...

    def __getitem__(self, idx):
        ...
        return img, target

    def __len__(self):
        ...
        return len(self.imgs)
```


## Defining your model
기존 모델을 수정하는 상황은 일반적으로 다음과 같다.

A. 사전훈련된 모델에서 시작하여 마지막 레이어를 미세조정한다.
B. 모델의 백본을 다른 모델의 것으로 교체한다.

본 튜토리얼에서는 Faster R-CNN에 기반하는 Mask R-CNN을 사용한다. 바운딩 박스와 클래스 점수를 예측하는 모델이다. 각 상황에서 모델을 구성하는 방법을 확인하고, 여기서는 데이터셋이 작아도 가능한 상황 A로 진행한다.

### 상황 A. Finetuning from a pretrained model
COCO에서 사전훈련된 모델에서 시작하여 특정 클래스에 대한 미세조정을 하는 상황이다. 클래스 수를 변경한 predictor를 다시 정의해서 기존 모델의 predictor와 교체했다.

``` py
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

### 상황 B. Modifying the model to add a different backbone
Mobilenet의 특징을 백본으로 가져온다. 이미지 사이즈를 맞춰주기 위해 anchor와 pooler를 설정한다. 이들을 모아 FasterRCNN 모델을 구성한다.

``` py
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
```

### An Instance segmentation model for PennFudan Dataset
상황 A 코드에는 box predictor만 설정되어 있다. Mask predictor를 추가한다.

``` py
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
```


## Putting everything together
[references/detection](https://github.com/pytorch/vision/tree/main/references/detection)에는 감지모델의 학습과 평가를 단순화하도록 돕는 다양한 함수가 제공된다. 이 폴더를 복사해서 사용...해야 하는데, 튜토리얼과 버전 차이가 커서 사용하기 어렵다.

### Testing `forward` method (Optional)
학습과 추론에서 모델은 (실제로는 더 복잡하지만) 다음과 같이 다룬다.

``` py
# Model and dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = PennFudanDataset(
    'PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

# For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
```

실제 메인함수는 다음과 같이 진행된다.

1. 장치 설정
2. 학습용/평가용 데이터셋 구성
3. 데이터셋에 데이터로더 연결
4. 모델 구성(헬퍼 함수 사용)
5. 모델을 장치에 연결
6. Optimizer 구성
7. 10 에포크 학습, 각 에포크마다 평가
