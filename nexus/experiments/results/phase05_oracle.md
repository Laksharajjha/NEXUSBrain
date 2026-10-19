# phase05_audit_oracle — oracle (20 seeds)

## Accuracy and calibration (mean ± std over seeds)

| split | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|
| train | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| dev | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| test_novel_combination | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| test_novel_word_in_known_frame | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| test_longer | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| test_contradiction | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |

## Support-stream checkpoints: test_novel_word_in_known_frame

| checkpoint | accuracy |
|---|---|
| before_support | 1.000 ± 0.000 |
| after_support | 1.000 ± 0.000 |

## Support-stream checkpoints: test_contradiction

| checkpoint | accuracy |
|---|---|
| before_stream | 1.000 ± 0.000 |
| before_corrections | 1.000 ± 0.000 |
| after_corrections | 1.000 ± 0.000 |

## Learning curves (dev)

| construction | examples-to-criterion (total seen) | retention: at stage end | retention: final |
|---|---|---|---|
| np_ask | 0 ± 0 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| np_assert | 0 ± 0 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| trans_ask | 0 ± 0 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| trans_assert | 0 ± 0 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Model size: 0.0 ± 0.0
