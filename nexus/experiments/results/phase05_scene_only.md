# phase05_audit_scene_only — scene_only (20 seeds)

## Accuracy and calibration (mean ± std over seeds)

| split | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|
| train | 0.530 ± 0.008 | 0.009 ± 0.006 | 0.249 ± 0.001 | 0.526 ± 0.006 |
| dev | 0.523 ± 0.021 | 0.025 ± 0.010 | 0.250 ± 0.001 | 0.527 ± 0.006 |
| test_novel_combination | 0.458 ± 0.034 | 0.072 ± 0.038 | 0.255 ± 0.003 | 0.529 ± 0.007 |
| test_novel_word_in_known_frame | 0.510 ± 0.035 | 0.041 ± 0.025 | 0.251 ± 0.003 | 0.525 ± 0.006 |
| test_longer | 0.539 ± 0.021 | 0.025 ± 0.016 | 0.248 ± 0.002 | 0.528 ± 0.006 |
| test_contradiction | 0.486 ± 0.098 | 0.097 ± 0.042 | 0.253 ± 0.007 | 0.528 ± 0.007 |

## Support-stream checkpoints: test_novel_word_in_known_frame

| checkpoint | accuracy |
|---|---|
| before_support | 0.501 ± 0.037 |
| after_support | 0.510 ± 0.035 |

## Support-stream checkpoints: test_contradiction

| checkpoint | accuracy |
|---|---|
| before_stream | 0.470 ± 0.096 |
| before_corrections | 0.445 ± 0.092 |
| after_corrections | 0.486 ± 0.098 |

## Learning curves (dev)

| construction | examples-to-criterion (total seen) | retention: at stage end | retention: final |
|---|---|---|---|
| np_ask | n/a | 0.509 ± 0.072 | 0.504 ± 0.055 |
| np_assert | n/a | 0.508 ± 0.051 | 0.518 ± 0.051 |
| trans_ask | n/a | 0.534 ± 0.050 | 0.534 ± 0.050 |
| trans_assert | n/a | 0.536 ± 0.047 | 0.536 ± 0.047 |

Model size: 17.0 ± 0.0
