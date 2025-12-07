# Vgg16 - Team Project

**Machine Vision Engineer** | 대한민국, 서울 | sejunkwon@outlook.com |
***

## 1. 레포지토리 설명
**Introduction**
* OpenCL 라이브러리를 이용한 Vggnet-16 CNN 뉴럴네트워크의 추론 최적화
* Weight 와 Bias 는 사전에 학습된 가중치를 불러와 사용
* 데이터셋은 CIFAR-10 데이터셋 사용
* 3000장의 32x32 이미지를 고정된 네트워크를 통해 CNN Forward, 이때 ML Runtime FrameWork를 사용하지 않고 직접 연산
* Parallel 연산을 통해 추론한 결과를 순차 처리 코드와 비교하여 올바르게 구현되었는지 검증

**Prerequisites**
* OpenCL과 CUDA 라이브러리 설치, C++ 빌드 및 실행이 가능한 환경
* 사용한 데이터셋의 다운로드 링크\
CIFAR-10 - <https://www.cs.toronto.edu/~kriz/cifar.html>\

**테스트 환경 - 실습실 Desktop**
* i7 8700K 6Core 12Threads
* 32 GB Main Memory
* RTX 3060 VRAM 12GB
***

## 2. 구현 과정

* CNN 순차처리 코드의 구조 파악
  - CNN레이어는 출력 feature map 의 크기를 유지해야할 때 padding이 필요함. 이것을 실제로 구현하기 위해 조건문을 통해 반복문 내부에서 인덱스가 마이너스인 경우 0으로 대체하는것을 이해함
  - 6중 for문에 의해 처리 시간이 기하급수적으로 늘어나는 것을 확인
* 최적화 작업 초기
  - 처음에는 for문을 unrolling 하여 반복문의 개수를 줄이는 것에 집중, 3000개의 이미지를 한번에 처리
  - 중간 발표 타 팀과 비교했을 때 턱없이 적은 성능 향상, 이를 통해 work item이 한번씩만 연산을 한다고 해도 개수가 너무 많으면 scheduling 되어 결국엔 순차적으로 실행된다는 것을 깨달음
* 개선 전략
  - ML Runtime framework에서는 CNN레이어에 대해 이미 연구로 검증된 최적화 기법들을 사용하는 것을 확인
  - Nvidia의 cuDNN 라이브러리에서는 CNN 에대해 GEMM, Winograd, FFT, im2col 기법으로 구현하는 것을 보고 강의시간에 배운 것과 관련있는 GEMM, im2col 기법을 사용하기로 결정
* 최적화 작업 중기
  - GEMM기법을 통해 Matrix Multiplification을 Parallel 처리로 구현 이때 유의미한 성능개선 체감
  - 하지만 이때도 강의에서 제시된 실행 시간의 Guidance의 상단에는 미치지 못함
* 최적화 작업 후기
  - 직관적으로 이해할 수 있는 im2col 기법을 이용해 주어진 이미지 행렬의 중복된 CNN연산부를 Colmn Vector로 펼침
  - 이를 GEMM Tiling 기법을 통해 더 빠르게 실행되도록 최적화함
  - 매 layer마다 똑같은 kernel 을 사용하는게 아닌 각 레이어의 입출력 크기에 맞게 커널을 다르게 작성 후 더욱 성능향상을 얻음
  - 이미지 3000장을 한번에 처리하는게 아닌 묶음 단위로 mini-batch 처리하여 성능향상을 얻음
***

## 3. 깨달은 점

* 구현
  - 초기에 스스로의 생각으로 구현할 때 OpenCL 커널함수를 실행할 때의 Constraints를 발견함
  - Global Work Item을 구성하는 Grid는 Work Group Size로 등분됨(같은 크기로 나눠짐)
  - Group 을 구성하는 Work Item 의 개수는 GPU 의 종류에 따라 다름 실습실 컴퓨터의 경우 제한은 1024개 였음
  - Group 내에서 Shared Memroy 의 크기 자체도 제한이 있음 구현 중에 커널이 실행될 때도 안 될 떄도 있었는데 Nvidia계열의 경우는 48KB로 알려짐
	- CNN 커널을 곱하고 input channel 에 대해 reduction 덧셈 병렬화를 할 때 발견, Group Size를 점진적으로 키우면서 여기에 쓰이는 Shared Memory의 크기 때문에 실행이 안 되었음
* 성능평가
  - 처음부터 끝까지 Work Item을 3차원으로 설정하여 진행함, 이는 이미지가 NCHW의 3개 채널로 들어오기 때문
  - 그러나 1등을 한 조는 모든 Work Item을 2차원으로 설정하여 진행, 이는 오버헤드를 줄이는 결과를 불러옴, 결과적으로 아쉽게 2등을 함
***