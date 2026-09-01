# AIMET Encodings

[English](../aimet-encodings.md) | 한국어

HNNX는 ONNX 모델과 AIMET encodings 0.6.1, 1.x, 2.x 형식을 함께 표시합니다.

## Encodings 불러오기

데스크톱 앱에서는 ONNX 모델과 encodings 파일을 함께 드래그합니다. 외부
`.data` sidecar도 같은 동작에 포함할 수 있습니다.

VS Code에서는 모델과 같은 폴더에서 다음 파일명을 자동으로 찾습니다.

```text
model.encodings
model.onnx.encodings
model.encodings.json
```

`hnnx.autoLoadEncodings` 설정에서 자동 불러오기를 제어합니다.

ONNX를 다시 열지 않고 File 메뉴에서 attachment를 관리합니다.

- **Load AIMET Encodings…**에서 파일을 선택합니다.
- **Reload Encodings**는 같은 로컬 또는 원격 URI를 다시 읽습니다.
- **Detach Encodings**는 첨부 파일만 제거합니다.
- `ENC` 도구 모음 버튼은 첨부 파일을 유지하면서 시각화만 켜거나 끕니다.

## 그래프 표기 해석

노드와 엔드포인트 배지는 명시적인 encoding 정보를 나타냅니다.

- `A8`, `A16`: 명시적 activation 또는 `Graph Input` QParam입니다.
- `W4`, `W8`, `W4/W8`: parameter bit width입니다.
- `A8→A16`: 입력 정밀도에서 명시적 출력 정밀도로 변경됩니다.
- `A8/A16→A16`: mixed 입력 정밀도가 명시적 A16 출력을 생성합니다.

전달된 정밀도는 그래프 edge에 `~A8`과 같은 흐린 레이블로 표시합니다. 따라서
정밀도 보존 경로와 명시적 QParam을 가진 노드를 구분할 수 있습니다.

배지, 텐서, 노드 또는 encoding이 있는 `Graph Input`을 클릭하면 scale,
offset/zero point, quantization range, axis, block size, granularity, symmetry
정보를 확인할 수 있습니다. HNNX는 legacy encoding의 명시적 min/max를 그대로
표시합니다. 정수 AIMET 1.x와 2.x 파일에 min/max가 없으면 bit width, scale,
offset/zero point로 표현 가능 범위를 계산하고 `range source`에 `explicit` 또는
`derived`를 표시합니다.

파생 범위는 계산 방식이 명확한 정수 per-tensor 및 per-channel encoding에만
적용합니다. 전달된 `~A8` 정밀도, floating-point encoding, 호환되지 않는 scale과
zero-point 배열, LPBQ scale 표현에는 범위를 임의로 생성하지 않습니다.

## 정밀도 전파

하나의 명확한 encoding 정밀도가 있으면 HNNX는 encoding이 없는 다음 연산을
지나서 정밀도를 추적합니다.

```text
Identity, Cast, Transpose, Reshape, Flatten, Squeeze, Unsqueeze,
DepthToSpace, SpaceToDepth, Split, Concat
```

AIMET 시각화에서 Cast는 정밀도 보존 연산으로 처리합니다. Concat activation
입력의 정밀도가 충돌하면 하나의 정밀도를 임의로 추론하지 않습니다.
TopK에서는 `values` 출력에만 activation 정밀도를 전달하고 `indices`는 ONNX
integer 텐서로 유지합니다.

이 기능은 화면에 표시할 정밀도만 추론합니다. Scale, zero-point 등의 QParam을
새로 만들지 않습니다.

## 통계 및 불일치 확인

Model information 화면은 다음 정보를 요약합니다.

- 전체 노드와 명시적 QParam이 있는 노드 수
- Activation 및 parameter bit-width 분포
- Mixed-precision 노드와 정밀도 전환
- 일치하거나 일치하지 않는 encoding 항목
- 추론된 activation 정밀도
- 검증 경고와 오류
- KV-cache 정밀도 분류

일치하지 않는 항목이 있다고 해서 항상 ONNX가 잘못된 것은 아닙니다. 내보내기
시점의 이름, 삭제된 텐서, 오래된 encodings 또는 의도적인 그래프 구조 편집이 원인일
수 있습니다. 처리 방법을 결정하기 전에 텐서 이름과 그래프 위치를 확인합니다.

## 편집 시 동작

AIMET encodings는 읽기 전용이며 GraphSurgeon 편집과 독립적입니다. ONNX 텐서의
이름, 존재 여부, 연결을 변경해도 외부 encodings 파일은 변경하지 않습니다. 복잡한
편집 중 기존 badge가 혼동을 줄 수 있다면 encodings를 분리하고, 편집된 ONNX에
맞게 다시 생성한 encoding 파일을 불러옵니다.

ONNX 편집 방법은 [그래프 편집](graph-editing.md)을 참고합니다.
