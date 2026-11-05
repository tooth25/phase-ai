# 합금 상 예측 AI (CALPHAD 학습 MLP)

조성·온도 → 평형 안정상 예측. 학습 데이터는 공개 열역학 DB(COST507, pycalphad 저장소 수록)로 평형 계산하여 생성.

## 현재 버전

- `phase_predictor.html`을 브라우저에서 열어 실행합니다.
- 경계 비교, 나란히 보기, 커튼, 차이 맵, 반투명 보기와 좌표 연동 확대·이동을 지원합니다.
- 주기율표 원소 선택·전체 해제, 조성·온도 입력과 그래프 선택 지점 마커를 제공합니다.
- 시스템의 라이트·다크 모드를 따릅니다.
- 차이 맵은 상 집합의 Jaccard 차이입니다. 화면의 기준 격자 일치율과 독립 시험 데이터 평가 성능은 구분해야 합니다.
- 모델은 기존 39개 합금계로 학습한 버전입니다. 신규 이원계 16개 데이터 추가·재학습은 아직 반영되지 않았습니다.

## 추가 수집 자료

함께 포함된 JSON 4개는 ORCHID 공개 계산 자료입니다. 현재 HTML의 모델·기준 격자에 연결되기 전의 수집본입니다.

| 파일 | 원본 평가 |
|---|---|
| `CuNi-07Tur.json` | Turchanin 외 (2007), DOI 10.1007/s11106-007-0013-9 |
| `CuMn-08He.json` | He 외 (2008), DOI 10.1016/j.jallcom.2007.03.041 |
| `FeNi-10Cac.json` | Cacciamani 외 (2010), DOI 10.1016/j.intermet.2010.02.026 |
| `CrNi-24Per.json` | Pereira dos Santos 외 (2024), DOI 10.1007/s11669-023-01079-3 |

출처 [ORCHID-Diagrams](https://data.orchid-project.org/), SGTE binary collection (Hallstedt 2025), CC BY 4.0. 원본 출처와 TDB SHA-256은 각 JSON의 `original_source`, `sha256` 필드에 있습니다. 모델 사용 시 원평가 논문도 함께 인용하세요.

## 구성
| 파일 | 역할 |
|---|---|
| `gen_data.py` | COST507로 이원계 31개·삼원계 8개 평형 계산 → `data/*.parquet` |
| `features.py` | 조성·온도 → 특성(몰분율, T/Tm, 조성가중 원자반경·전기음성도·VEC·융점, 원자크기 불일치 δ) |
| `train.py` | MLP(256-256-128) 다중 라벨 학습, 무작위 분할·삼원계 hold-out 평가 → `model.joblib`, `report.json` |
| `predict.py` | 예측 CLI, 이원 예측 상태도 그림 |
| `export_web.py` | 웹 데모용 가중치·기준 격자 내보내기 |
| `model.npz` | 학습된 모델 가중치 (numpy 형식, scikit-learn 버전과 무관) |
| `phase_predictor.html` | 더블클릭으로 여는 단독 실행 데모 (설치 불필요, 오프라인 동작) |

## 사용
가장 간단한 방법: `phase_predictor.html`을 브라우저(Chrome·Edge)로 열기.

명령줄 예측은 numpy만 있으면 됨:
```bash
pip install numpy matplotlib
python predict.py "Cu-5Al-2Si" 600          # at%, 첫 원소 잔부, °C
python predict.py "Al-4.5Cu" 500 --wt        # wt%
python predict.py "Al-10Cu" --map CU         # 예측 상태도 png
```
재학습(추가 패키지 필요: `pip install -r requirements.txt`): `python gen_data.py 4 && python train.py && python export_web.py` (2코어 기준 데이터 약 30분, 학습 약 40분)

## 성능
- 학습 계 내부(무작위 20%): 상 집합 완전일치 92.8%, Jaccard 0.97
- 미학습 삼원계(Al-Cu-Si / Al-Cu-Mg / Cu-Mg-Zn 각각 제외): 완전일치 38~48%, Jaccard 0.62~0.68

## 한계
- Ga·Ge 등 COST507 미수록 원소는 학습 데이터가 없음. Cu-Ga-Si, Cu-Ge-Al에 적용하려면 해당 계 TDB(문헌 평가값)를 `gen_data.py`에 추가해 재학습해야 함.
- 학습 라벨에 없는 상(미학습 삼원 화합물 등)은 예측 불가.
- 평형 상만 다룸(준안정 석출상·속도론 미반영).
- 계산 격자(이원 1 at% × 20 K)보다 세밀한 경계 위치는 보장되지 않음.
