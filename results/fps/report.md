# Frame-rate robustness results

## invariance_01_0026
| run | frames | spearman | pearson | ks | median_ratio | alarm_jaccard | box_jaccard | ref_alarm_frames | run_alarm_frames | ref_box_frames | run_box_frames | event_matches | first_alarm_delta_s | run_auroc | run_f1 | run_fa_per_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fps5 | 252.000 | 0.983 | 0.992 | 0.048 | 1.000 | 0.938 | 0.231 | 77.000 | 80.000 | 11.000 | 5.000 | 76.000 | -0.400 | 0.891 | 0.657 | 40.637 |
| fps20 | 503.000 | 0.999 | 1.000 | 0.010 | 1.000 | 0.994 | 1.000 | 155.000 | 156.000 | 20.000 | 20.000 | 3.000 | 0.000 | 0.883 | 0.652 | 78.884 |
| fps30 | 503.000 | 0.999 | 1.000 | 0.010 | 1.000 | 0.987 | 1.000 | 155.000 | 157.000 | 20.000 | 20.000 | 3.000 | 0.000 | 0.883 | 0.650 | 80.080 |

## invariance_03_0061
| run | frames | spearman | pearson | ks | median_ratio | alarm_jaccard | box_jaccard | ref_alarm_frames | run_alarm_frames | ref_box_frames | run_box_frames | event_matches | first_alarm_delta_s | run_auroc | run_f1 | run_fa_per_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fps5 | 120.000 | 0.996 | 0.997 | 0.042 | 1.002 | 0.991 | 0.839 | 110.000 | 111.000 | 56.000 | 58.000 | 110.000 | 0.000 | 0.740 | 0.356 | 219.328 |

## invariance_avenue_01
| run | frames | spearman | pearson | ks | median_ratio | alarm_jaccard | box_jaccard | ref_alarm_frames | run_alarm_frames | ref_box_frames | run_box_frames | event_matches | first_alarm_delta_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fps10 | 299.000 | 0.987 | 0.994 | 0.067 | 1.003 | 1.000 | 0.765 | 299.000 | 299.000 | 199.000 | 214.000 | 299.000 | 0.000 |

## quantisation_01_0026
| run | frames | spearman | pearson | ks | median_ratio | alarm_jaccard | box_jaccard | ref_alarm_frames | run_alarm_frames | ref_box_frames | run_box_frames | event_matches | first_alarm_delta_s | run_auroc | run_f1 | run_fa_per_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dt0p05 | 503.000 | 0.970 | 0.967 | 0.203 | 0.953 | 0.845 | 0.000 | 155.000 | 131.000 | 20.000 | 4.000 | 1.000 | 0.800 | 0.897 | 0.661 | 57.256 |
