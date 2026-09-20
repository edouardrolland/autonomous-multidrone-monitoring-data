# Reproduced statistics

Every number below was recomputed from `data/` by the scripts in this directory.

## PSO tuning (Table: specialist hyperparameter sets)

| regime | best trial | w | c1 | c2 | c1/c2 | median Omega |
|---|---|---|---|---|---|---|
| grazing | 70 | 0.68 | 2.28 | 0.69 | 3.3 | 0.256 |
| travelling | 96 | 0.61 | 2.45 | 0.53 | 4.6 | 0.285 |
| panic | 55 | 0.67 | 2.11 | 0.67 | 3.2 | 0.195 |

## PSO validation (paired on the 108 configuration means per regime)

| regime | vs | n | win | median diff | Wilcoxon p | Holm p | TOST p | equivalent |
|---|---|---|---|---|---|---|---|---|
| grazing | travelling | 108 | 0.47 | -0.0005 | 0.741 | 1 | 4.38e-14 | yes |
| grazing | panic | 108 | 0.53 | +0.0004 | 0.715 | 1 | 1.67e-16 | yes |
| grazing | field | 108 | 0.61 | +0.0011 | 0.0107 | 0.096 | 9.31e-12 | yes |
| grazing | generalist | 108 | 1.00 | +0.0267 | 1.87e-19 | - | 1 | no |
| travelling | grazing | 108 | 0.63 | +0.0013 | 0.0163 | 0.114 | 8.98e-06 | yes |
| travelling | panic | 108 | 0.59 | +0.0013 | 0.33 | 1 | 2.2e-09 | yes |
| travelling | field | 108 | 0.60 | +0.0021 | 0.0114 | 0.096 | 8.22e-05 | yes |
| travelling | generalist | 108 | 1.00 | +0.0349 | 1.87e-19 | - | 1 | no |
| panic | grazing | 108 | 0.54 | +0.0002 | 0.622 | 1 | 1.36e-20 | yes |
| panic | travelling | 108 | 0.57 | +0.0008 | 0.055 | 0.33 | 1.62e-20 | yes |
| panic | field | 108 | 0.50 | +0.0000 | 0.998 | 1 | 2.12e-21 | yes |
| panic | generalist | 108 | 0.99 | +0.0216 | 1.98e-19 | - | 1 | no |

Tuned vs untuned default: grazing +11.3%, travelling +14.4%, panic +11.8%

## Scalability

- 22500 runs over 4500 (herd, M) cells
- t_ref = 6.2 s; normalised solve cost spans 1.0-11.7
- mean solve 26.1 s over the grid; largest cell 72.0 s
- dispersion: normalised mean cost changes -26% from sigma = 10 m to 100 m

## Field perception (Table: perception accuracy)

| mission | N_GT | recall | precision | median position error | median body-axis error |
|---|---|---|---|---|---|
| F1 | 188 | 97% | 97% | 0.23 m | 11.6 deg |
| F2 | 754 | 79% | 91% | 0.46 m | 16.8 deg |
| F3 | 519 | 65% | 99% | 0.30 m | 12.4 deg |
| F4 | 409 | 93% | 93% | 0.25 m | 11.1 deg |
| pooled | 1870 | 80% | 94% | 0.32 m | 13.0 deg |

## Field coverage of the surfaces of interest

Pooled over the 215 monitoring frames of the four missions:

| surface | V_k >= 0.5 | V_k >= 0.75 | V_k >= 0.9 |
|---|---|---|---|
| dorsal | 98.6% | 96.7% | 96.7% |
| left flank | 93.5% | 75.8% | 46.5% |
| right flank | 87.0% | 71.6% | 35.8% |

| mission | median dorsal | median left flank | median right flank |
|---|---|---|---|
| F1 | 1.00 | 0.75 | 0.75 |
| F2 | 1.00 | 0.81 | 0.82 |
| F3 | 1.00 | 0.92 | 0.92 |
| F4 | 1.00 | 1.00 | 0.93 |

## Stand-off compliance

- minimum stand-off respected at 85.9% of 489 evaluated drone-animal instances
- median deficit among breaches 1.1 m
