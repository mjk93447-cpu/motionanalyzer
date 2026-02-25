# 참고 논문 — 논문 작성용

DREAM·PatchCore 앙상블, Normal FP 개선, 정확도 개선 관련 논문 분석 요약. 참고문헌에 추가하여 인용.

---

## 1. DRAEM — Discriminative Reconstruction Anomaly Embedding

**저자**: Zavrtanik, V., Kristan, M., Skočaj, D.  
**출처**: ICCV 2021  
**arXiv**: [2108.07610](https://arxiv.org/abs/2108.07610)  
**코드**: https://github.com/VitjanZ/DRAEM

### 요약

- **접근**: 재구성(Reconstruction) + 판별(Discriminative) 결합. 기존 생성형 모델은 재구성 오차만 사용하고 후처리가 필요했으나, DRAEM은 **이상/정상 판별을 직접 학습**.
- **학습**: 정상 이미지 + 합성 이상 시뮬레이션으로 (입력, 이상 제거 재구성) 쌍을 만들고, 판별자가 정상/이상을 구분하도록 학습.
- **성능**: MVTec AD에서 image-level ROC AUC 98.1%, pixel-wise ROC AUC 97.5%. 비감독 방법 대비 크게 우수, 일부 감독 방법에 근접.
- **FPCB 관련점**: DREAM은 DRAEM 전략을 적용. 테이블/시계열 특징에 맞게 적응. temporal 패턴·재구성 오차로 정상과 이상 구분.

### BibTeX

```bibtex
@inproceedings{zavrtanik2021draem,
  title={{DRAEM} - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection},
  author={Zavrtanik, Vitjan and Kristan, Matej and Sko{\v{c}}aj, Danijel},
  booktitle={Proc. ICCV},
  pages={8330--8339},
  year={2021}
}
```

---

## 2. PatchCore — Towards Total Recall in Industrial Anomaly Detection

**저자**: Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P.  
**출처**: CVPR 2022  
**arXiv**: [2106.08265](https://arxiv.org/abs/2106.08265)  
**코드**: https://github.com/amazon-research/patchcore-inspection

### 요약

- **접근**: 정상 패치 특징의 **대표 메모리 뱅크** + k-NN 거리 기반 이상 스코어. ImageNet 사전학습 CNN으로 패치 특징 추출.
- **핵심**: Coreset 샘플링으로 메모리 뱅크 축소, 추론 시간 유지. 중간층 특징(ResNet block 2–3)으로 텍스처·형상 균형.
- **성능**: MVTec AD image-level AUROC 99.6%, 기존 SOTA 대비 오차 절반 수준. Detection과 localization 모두 우수.
- **FP·정확도**: Cold-start(정상만 학습) 설정에서 FP–FN 트레이드오프 관리. Total recall 지향 설계로 recall 극대화, precision도 유지.
- **FPCB 관련점**: PatchCore를 이미지가 아닌 motion 특징 벡터에 적용. DREAM과 상호 보완적 오류 패턴 → AND 앙상블로 FP 상호 필터링.

### BibTeX

```bibtex
@inproceedings{roth2022patchcore,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle={Proc. CVPR},
  year={2022}
}
```

---

## 3. A Dual-Branch Ensemble Learning Method for Industrial Anomaly Detection

**저자**: Cai, J., Wu, Z., Hua, R., Mao, S., Zhang, Y., Guo, R., Lin, K.  
**출처**: Applied Sciences (MDPI), 2026, 16(3), 1597  
**DOI**: https://doi.org/10.3390/app16031597

### 요약

- **접근**: PCA(선형 전역 구조) + Scattering transform(다중 스케일 텍스처)의 **쌍분기** 특징 + **이질적 앙상블** (SVM, RF, XGBoost, LightGBM 등).
- **퓨전**: Stacking 메타학습자(LR, XGBoost, LightGBM)로 결정 수준 퓨전. Quantile 기반 임계값 탐색, 확률 보정으로 안정적 출력.
- **성능**: MVTec AD, BTAD에서 단일 PCA 모델 대비 F1 약 31%, 26% 향상. AUC 0.91(MVTec), 0.96(BTAD).
- **FP·정확도**: 이질적 모델 결합으로 견고성·일반화 향상, FP 완화. 제한된 labeled anomaly에서 실용적.
- **FPCB 관련점**: DREAM+PatchCore 이질적 앙상블과 유사. AND 규칙으로 보수적 결정, FP 필터링 전략 참고 가능.

### BibTeX

```bibtex
@article{cai2026dual,
  title={A Dual-Branch Ensemble Learning Method for Industrial Anomaly Detection: Fusion and Optimization of Scattering and PCA Features},
  author={Cai, Jing and Wu, Zhuo and Hua, Runan and Mao, Shaohua and Zhang, Yulun and Guo, Ran and Lin, Ke},
  journal={Applied Sciences},
  volume={16},
  number={3},
  pages={1597},
  year={2026},
  publisher={MDPI}
}
```

---

## 4. Multi-Model Anomaly Detection with Dynamic Loss Weighting and Soft-Hard Features Loss

**저자**: (Springer Neural Computing and Applications)  
**출처**: Springer, 2025  
**링크**: https://link.springer.com/article/10.1007/s00521-025-11367-3

### 요약

- **접근**: **다중 모델** + **동적 손실 가중치** + **soft-hard 특징 손실**.
- **동적 손실 가중치**: 각 모델 기여도를 학습 중 조정, 서로 다른 전문성을 가진 네트워크가 협력하도록 설계.
- **Soft-Hard 특징 손실**: 미세 이상 영역을 더 잘 포착. 고오차 특징 강조, 저오차 특징도 활용해 소형 결함 검출.
- **성능**: MVTec AD에서 AU-PRO +0.6%, VisA에서 AU-ROC +0.4% 향상. 정밀 localization과 양적 성능 모두 개선.
- **FPCB 관련점**: 다중 모델(DREAM, PatchCore) 앙상블과 일치. micro_crack 등 미세 결함에 대한 soft-hard 손실 아이디어는 임계값·스코어 설계 참고 가능.

### BibTeX

```bibtex
@article{springer2025multimodel,
  title={Multi-Model Anomaly Detection for Industrial Inspection with Dynamic Loss Weighting and Soft-Hard Features Loss},
  journal={Neural Computing and Applications},
  year={2025},
  publisher={Springer},
  note={https://doi.org/10.1007/s00521-025-11367-3}
}
```

---

## 5. Beyond Academic Benchmarks: Critical Analysis for Visual Industrial Anomaly Detection (2025)

**저자**: (arXiv 2503.23451)  
**요약**: 학술 벤치마크만으로는 실제 산업 환경 성능을 반영하기 어렵다는 점을 지적. Precision–Recall 트레이드오프와 FP 관리가 실제 배포에서 중요함을 강조.  
**FPCB 관련점**: 합성 데이터 기반 개발 후 실제 데이터 전이 시 고려사항으로 인용 가능.

---

## 논문 작성 시 인용 가이드

| 인용 시점 | 논문 |
|-----------|------|
| DREAM/DRAEM 방법 설명 | [1] Zavrtanik et al. |
| PatchCore 방법 설명 | [2] Roth et al. |
| 앙상블·FP 감소 논거 | [3] Cai et al., [4] Springer multi-model |
| MVTec AD 벤치마크 | [3] Bergmann et al. |
| 합성+실제 전이 | [4] ISP-AD (기존), [5] Beyond Academic Benchmarks (선택) |

---

## 6. Iterative Refinement / 개발 루프 (문헌)

| 논문 | 요약 |
|------|------|
| **Self-Refine** (NeurIPS 2023) | 모델이 자체 피드백으로 반복 개선; ~20% 향상 |
| **Verifier-Guided Refinement** (arXiv 2504.01931) | 검증자 가이드 반복; 코드/텍스트 3~6% 향상 |
| **NeurIPS 2019 Reproducibility** | 코드·체크리스트·재현성 프로그램 |
| **ML Experimentation** (arXiv 2511.21354) | 실험 설계·문서화·검증 rigor |

**적용**: 상황분석→개선→테스트→평가→추가개발 루프, gap-driven, 과잉 보정 방지.

---

**문서 버전**: 1.0 | **최종 갱신**: 2026-02-23
