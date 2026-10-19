# phase05_audit_sentence_only — sentence_only (20 seeds)

## Accuracy and calibration (mean ± std over seeds)

| split | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|
| train | 0.529 ± 0.004 | 0.014 ± 0.005 | 0.249 ± 0.000 | 0.516 ± 0.003 |
| dev | 0.500 ± 0.027 | 0.029 ± 0.020 | 0.250 ± 0.001 | 0.517 ± 0.003 |
| test_novel_combination | 0.495 ± 0.025 | 0.032 ± 0.019 | 0.251 ± 0.001 | 0.519 ± 0.004 |
| test_novel_word_in_known_frame | 0.505 ± 0.025 | 0.024 ± 0.013 | 0.250 ± 0.001 | 0.515 ± 0.003 |
| test_longer | 0.510 ± 0.030 | 0.031 ± 0.016 | 0.250 ± 0.002 | 0.522 ± 0.008 |
| test_contradiction | 0.506 ± 0.029 | 0.030 ± 0.021 | 0.251 ± 0.001 | 0.521 ± 0.008 |

## Support-stream checkpoints: test_novel_word_in_known_frame

| checkpoint | accuracy |
|---|---|
| before_support | 0.505 ± 0.025 |
| after_support | 0.505 ± 0.025 |

## Support-stream checkpoints: test_contradiction

| checkpoint | accuracy |
|---|---|
| before_stream | 0.502 ± 0.019 |
| before_corrections | 0.496 ± 0.028 |
| after_corrections | 0.506 ± 0.029 |

## Learning curves (dev)

| construction | examples-to-criterion (total seen) | retention: at stage end | retention: final |
|---|---|---|---|
| np_ask | n/a | 0.505 ± 0.031 | 0.505 ± 0.048 |
| np_assert | n/a | 0.507 ± 0.059 | 0.492 ± 0.056 |
| trans_ask | n/a | 0.506 ± 0.041 | 0.506 ± 0.041 |
| trans_assert | n/a | 0.495 ± 0.044 | 0.495 ± 0.044 |

Model size: 197.0 ± 0.0
