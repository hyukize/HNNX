# AIMET Encodings

[English](../aimet-encodings.md) | 한국어

HNNX는 ONNX 모델과 AIMET encodings 0.6.1, 1.x, 2.x 형식을 함께 표시합니다.

## Encodings 불러오기

데스크톱 앱에서는 ONNX 모델과 encodings 파일을 함께 드래그합니다. External
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
- **Reload Encodings**는 같은 local 또는 remote URI를 다시 읽습니다.
- **Detach Encodings**는 attachment만 제거합니다.
- `ENC` toolbar 버튼은 attachment를 유지하면서 시각화만 켜거나 끕니다.

## 그래프 표기 해석

Node와 endpoint badge는 명시적인 encoding 정보를 나타냅니다.

- `A8`, `A16`: 명시적 activation 또는 Graph Input QParam입니다.
- `W4`, `W8`, `W4/W8`: parameter bit width입니다.
- `A8→A16`: input precision에서 명시적 output precision으로 변경됩니다.
- `A8/A16→A16`: mixed input precision이 명시적 A16 output을 생성합니다.

전달된 precision은 graph edge에 `~A8`과 같은 흐린 label로 표시합니다. 따라서
precision-preserving 경로와 명시적 QParam을 가진 노드를 구분할 수 있습니다.

Badge, tensor, node 또는 encoding이 있는 Graph Input을 클릭하면 encoding 파일에
존재하는 scale, offset/zero point, min/max, axis, block size, granularity,
symmetry 정보를 확인할 수 있습니다.

## Precision 전파

하나의 명확한 encoded precision이 있으면 HNNX는 encoding이 없는 다음 연산을
지나서 precision을 추적합니다.

```text
Identity, Cast, Transpose, Reshape, Flatten, Squeeze, Unsqueeze,
DepthToSpace, SpaceToDepth, Split, Concat
```

AIMET 시각화에서 Cast는 precision-preserving으로 처리합니다. Concat activation
input의 precision이 충돌하면 하나의 precision을 임의로 추론하지 않습니다.
TopK에서는 `values` output에만 activation precision을 전달하고 `indices`는 ONNX
integer tensor로 유지합니다.

이 기능은 화면에 표시할 precision만 추론합니다. Scale, zero-point 등의 QParam을
새로 만들지 않습니다.

## 통계 및 mismatch 확인

Model information 화면은 다음 정보를 요약합니다.

- 전체 node와 명시적 QParam이 있는 node 수
- Activation 및 parameter bit-width 분포
- Mixed-precision node와 transition
- 일치하거나 일치하지 않는 encoding entry
- 추론된 activation precision
- Validation warning과 error
- KV-cache precision 분류

일치하지 않는 entry가 있다고 해서 항상 ONNX가 잘못된 것은 아닙니다. Export
시점의 이름, 삭제된 tensor, 오래된 encodings 또는 의도적인 topology 편집이 원인일
수 있습니다. 처리 방법을 결정하기 전에 tensor 이름과 graph 위치를 확인합니다.

## 편집 시 동작

AIMET encodings는 read-only이며 GraphSurgeon 편집과 독립적입니다. ONNX tensor를
rename, delete, rewire해도 외부 encodings 파일은 변경하지 않습니다. 복잡한
편집 중 기존 badge가 혼동을 줄 수 있다면 encodings를 분리하고, 편집된 ONNX에
맞게 다시 생성한 encoding 파일을 불러옵니다.

ONNX 편집 방법은 [그래프 편집](graph-editing.md)을 참고합니다.
