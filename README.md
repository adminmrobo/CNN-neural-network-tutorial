# 하루의 실험실 : 필터를 미끄러뜨리는 망 (CNN)

**AHNBOAT LAB · 조수 하루의 3차원 실험실 · EP.03** — 랩타운 2층 실험 방의 실험대 (4, 0, 16): https://adminmrobo.github.io/ahn_boat_lab_town/haru3d.html

실험조수 하루가 CNN 실험을 직접 다시 돌리고, 숫자와 그림으로 설명합니다.

- 페이지: https://adminmrobo.github.io/CNN-neural-network-tutorial/
- 연구실 홈: https://www.ahnboat.xyz/

## 무엇을 하나

질문은 하나입니다. **왼쪽에서만 본 고양이를 오른쪽으로 옮겨도 알아볼까?**

16×16 회색 그림(0~255)으로 고양이·개·여우 얼굴을 코드로 만들었습니다. 학습 그림 1,800장은 얼굴을 왼쪽(dx 0~2)에만 두었고, 시험 그림은 두 묶음입니다. 하나는 같은 왼쪽에 둔 처음 보는 그림 900장, 다른 하나는 오른쪽(dx 4~5)으로 옮긴 그림 900장입니다.

같은 데이터로 두 망을 학습시켰습니다. 둘 다 모멘텀 0.9 경사하강법, 학습률 0.05, 묶음 60장으로 40바퀴(에포크)를 돌렸습니다.

- **일반 망**: 256 → 32 (ReLU) → 3
- **CNN**: 3×3 필터 4개 (ReLU) → 2×2 최댓값 줄이기 → 3×3 필터 8개 (ReLU) → 지도마다 최댓값 → 3

| | 일반 망 | CNN |
|---|---:|---:|
| 왼쪽 그림 정답률 (처음 보는 900장) | 100.0 % | 100.0 % |
| **옮긴 그림 정답률 (900장)** | **25.8 %** (0.2578) | **98.0 %** (0.9800) |
| 가중치 수 | 8,323 | 363 |
| 그림 한 장의 곱셈 수 | 8,288 | 27,672 |
| 같은 고양이를 3픽셀 옮겼을 때 | 개 99.9 % (0.9985) | 고양이 99.8 % (0.9983) |

핵심 숫자는 **25.8 % · 98.0 % · 23배**(더 적은 가중치, 8,323 ÷ 363 = 22.9)입니다. 대신 CNN은 필터를 모든 위치에 되풀이해 곱하므로 곱셈이 3.3배 많습니다.

학습 그림에서 얼굴이 지나간 열은 0~12번뿐이고, 오른쪽 끝 13~15번 열은 한 번도 밝아진 적이 없습니다(`cover_train`). 일반 망은 픽셀마다 따로 가중치를 붙이므로, 처음 밝아지는 위치에서 답이 무너집니다. CNN은 같은 필터 숫자를 그림 전체에 나눠 쓰고, 마지막에 지도마다 가장 큰 값만 남깁니다. 그래서 위치가 바뀌어도 비슷한 숫자가 나옵니다.

## 페이지 구성 (10개 파트, 대사 37개)

1. 인트로 · 2. 필터 미끄러뜨리기 · 3. 일반 망의 문제 · 4. 필터 4개, 지도 4장 · 5. 학습 중: 필터가 자란다 · 6. 층을 쌓으면 (3D) · 7. Fukushima 1980 · LeCun 1989 (3D) · 8. 코드: 묶고, 곱하고, 최댓값 · 9. 결과: 25.8% 대 98.0% · 10. 정리와 대가

장면은 모두 캔버스로 그립니다. 특징 지도, 최댓값 8개, 고양이 점수는 학습이 끝난 CNN 가중치(`cnn.K1·K2·V`)로 페이지 안에서 직접 계산합니다. 페이지 계산과 파이썬 결과의 확률 차이는 0.0001 미만입니다. 3D 장면은 x(오른쪽, `#e4572e`)·y(위, `#2a9d6f`)·z(앞, `#1f8fd6`) 오른손 좌표계 위에 그림(y = 0), 지도 4장(y = 6), 지도 8장(y = 12)을 쌓았습니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `index.html` | 완성 페이지 (한 파일, 외부 요청 없음) |
| `data/cnn_run.json` | 실험 기록: 바퀴별 손실·정답률, 바퀴별 1층 필터, 최종 CNN 가중치, 고양이 옮기기 점수, 픽셀별 얼굴 지나간 비율 |
| `data/haru_lines.json` | 하루의 대사 [화면용, 읽기용] |
| `data/haru_sprites.json` | 하루 표정 7장 (WebP, 216px) |
| `tools/experiment.py` | 실험 재현 (numpy, 시드 7) |
| `tools/haru_sprites.py` | 캐릭터 팩에서 표정 그림 만들기 |
| `tools/page_template.html` | 페이지 틀 (`__DATA__` · `__SCRIPT__` · `__HARU__`) |
| `tools/build_page.py` | 틀 + 데이터 → `index.html` |

다시 만들기 (저장소 맨 위 폴더에서):

```bash
python3 tools/experiment.py     # 약 7초, 두 번 돌려도 같은 파일
python3 tools/haru_sprites.py   # ../fruitfly-lab-expressions 가 있을 때만
python3 tools/build_page.py
```

## 출처

- 실험 아이디어: 16×16 고양이 그림 하나로 합성곱 신경망을 설명한 짧은 영상(“AI sees 256 numbers … That's a convolutional neural network.”). 원본 영상·소리·그림은 쓰지 않았습니다. 그림은 코드로 새로 만들었고, 영상 끝의 예시 출력(cat 94 %)은 쓰지 않았습니다. 대신 실제로 학습시킨 망의 숫자를 씁니다.
- K. Fukushima, *Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position*, Biological Cybernetics 36 (1980).
- Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel, *Backpropagation Applied to Handwritten Zip Code Recognition*, Neural Computation 1(4): 541–551 (1989), doi:10.1162/neco.1989.1.4.541.
- 하루 그림: [fruitfly-lab-expressions](https://adminmrobo.github.io/fruitfly-lab-expressions/) 캐릭터 팩. 쓴 표정은 E07 설명, E01 기쁨, E08 안내, E04 놀람, E09 생각, E12 응원, E10 확인입니다.

Reon Ahn's Page © 2022 Reon Ahn · CC BY-NC 4.0
