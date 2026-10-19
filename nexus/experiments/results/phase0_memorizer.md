# phase0_harness_check — memorizer (20 seeds)

## Accuracy and calibration (mean ± std over seeds)

| split | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|
| train | 1.000 ± 0.000 | 0.333 ± 0.000 | 0.111 ± 0.000 | 0.667 ± 0.000 |
| dev | 0.493 ± 0.021 | 0.049 ± 0.015 | 0.254 ± 0.003 | 0.536 ± 0.004 |
| test_novel_combination | 0.500 ± 0.000 | 0.006 ± 0.004 | 0.250 ± 0.000 | 0.506 ± 0.004 |
| test_novel_word_in_known_frame | 0.500 ± 0.000 | 0.006 ± 0.004 | 0.250 ± 0.000 | 0.506 ± 0.004 |
| test_longer | 0.500 ± 0.000 | 0.006 ± 0.004 | 0.250 ± 0.000 | 0.506 ± 0.004 |
| test_contradiction | 0.500 ± 0.000 | 0.005 ± 0.005 | 0.250 ± 0.000 | 0.505 ± 0.005 |

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
| np_ask | n/a | 0.508 ± 0.048 | 0.508 ± 0.048 |
| np_assert | n/a | 0.489 ± 0.052 | 0.489 ± 0.052 |
| trans_ask | n/a | 0.479 ± 0.041 | 0.479 ± 0.041 |
| trans_assert | n/a | 0.497 ± 0.022 | 0.497 ± 0.022 |

Model size: 2021.7 ± 19.1
