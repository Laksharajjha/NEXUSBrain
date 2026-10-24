# phase06_audit_sentence_only — sentence_only (20 seeds)

## Accuracy and calibration (mean ± std over seeds)

| split | n | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|---|
| train | 1600 | 0.500 ± 0.000 | 0.002 ± 0.000 | 0.250 ± 0.000 | 0.502 ± 0.000 |
| dev | 400 | 0.500 ± 0.000 | 0.002 ± 0.000 | 0.250 ± 0.000 | 0.502 ± 0.000 |
| test_novel_combination | 400 | 0.500 ± 0.000 | 0.002 ± 0.000 | 0.250 ± 0.000 | 0.502 ± 0.000 |
| test_novel_word_in_known_frame | 200 | 0.500 ± 0.000 | 0.003 ± 0.001 | 0.250 ± 0.000 | 0.503 ± 0.001 |
| test_longer | 300 | 0.500 ± 0.000 | 0.005 ± 0.001 | 0.250 ± 0.000 | 0.505 ± 0.001 |
| test_contradiction | 200 | 0.500 ± 0.000 | 0.002 ± 0.001 | 0.250 ± 0.000 | 0.502 ± 0.001 |

## Support-stream checkpoints: test_novel_word_in_known_frame

| checkpoint | accuracy |
|---|---|
| before_support | 0.500 ± 0.000 |
| after_support | 0.500 ± 0.000 |

## Support-stream checkpoints: test_contradiction

| checkpoint | accuracy |
|---|---|
| before_stream | 0.500 ± 0.000 |
| before_corrections | 0.500 ± 0.000 |
| after_corrections | 0.500 ± 0.000 |

## Learning curves (dev)

| construction | examples-to-criterion (total seen) | retention: at stage end | retention: final |
|---|---|---|---|
| np_ask | n/a | 0.500 ± 0.000 | 0.500 ± 0.000 |
| np_assert | n/a | 0.500 ± 0.000 | 0.500 ± 0.000 |
| trans_ask | n/a | 0.500 ± 0.000 | 0.500 ± 0.000 |
| trans_assert | n/a | 0.500 ± 0.000 | 0.500 ± 0.000 |

Model size: 197.0 ± 0.0
