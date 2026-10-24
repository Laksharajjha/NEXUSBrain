# phase06_audit_stubborn_oracle — stubborn_oracle (20 seeds)

## Accuracy and calibration (mean ± std over seeds)

| split | n | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|---|
| train | 1600 | 0.925 ± 0.008 | 0.075 ± 0.008 | 0.075 ± 0.008 | 1.000 ± 0.000 |
| dev | 400 | 0.931 ± 0.010 | 0.068 ± 0.010 | 0.069 ± 0.010 | 1.000 ± 0.000 |
| test_novel_combination | 400 | 0.907 ± 0.037 | 0.093 ± 0.037 | 0.093 ± 0.037 | 1.000 ± 0.000 |
| test_novel_word_in_known_frame | 200 | 0.908 ± 0.024 | 0.092 ± 0.024 | 0.092 ± 0.024 | 1.000 ± 0.000 |
| test_longer | 300 | 0.790 ± 0.019 | 0.210 ± 0.019 | 0.210 ± 0.019 | 1.000 ± 0.000 |
| test_contradiction | 200 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Support-stream checkpoints: test_novel_word_in_known_frame

| checkpoint | accuracy |
|---|---|
| before_support | 0.908 ± 0.024 |
| after_support | 0.908 ± 0.024 |

## Support-stream checkpoints: test_contradiction

| checkpoint | accuracy |
|---|---|
| before_stream | 0.000 ± 0.000 |
| before_corrections | 0.000 ± 0.000 |
| after_corrections | 0.000 ± 0.000 |

## Learning curves (dev)

| construction | examples-to-criterion (total seen) | retention: at stage end | retention: final |
|---|---|---|---|
| np_ask | 0 ± 0 (19/20 seeds) | 0.946 ± 0.028 | 0.946 ± 0.028 |
| np_assert | 0 ± 0 (19/20 seeds) | 0.952 ± 0.029 | 0.952 ± 0.029 |
| trans_ask | 0 ± 0 (16/20 seeds) | 0.913 ± 0.025 | 0.913 ± 0.025 |
| trans_assert | 0 ± 0 (17/20 seeds) | 0.915 ± 0.028 | 0.915 ± 0.028 |

Model size: 0.0 ± 0.0
