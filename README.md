# Pytorch Tutorial
## 참고
1. Introduction to PyTorch
    - [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
2. Image and Video
    - [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
    - [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)


## 환경
> Pytorch Build: Stable(1.11.0), OS: Linux, Package: Pip, Language: Python, Compute Platform: CUDA 11.3

1. 그래픽카드는 CUDA를 설치하면 자동으로 설치된다.
2. `https://pytorch.org/get-started/locally`에서 설치에 필요한 CUDA 버전을 확인한다. 그리고 `https://developer.nvidia.com/cuda-toolkit-archive`에서 안내에 따라 설치한다. 단, 마지막 명령어에 버전을 명시해야 해당 버전으로 설치된다: `sudo apt-get -y install cuda-11-3`
3. `https://developer.nvidia.com/rdp/cudnn-download`에서 설치한 CUDA를 지원하는 cuDNN 버전을 찾아 설치한다.
4. `https://pytorch.org/get-started/locally`에서 Pytorch를 설치한다.
5. 이하의 코드로 테스트한다.
    ``` py
    import torch
    print(torch.rand(5, 3))
    print(torch.cuda.is_available())
    ```
