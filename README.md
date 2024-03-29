# Robust-methods-for-scale-change-detection
The code is an additional material for submitted publication named "Robust variance estimators in application  to segmentation of measurement data distorted by impulsive and non-Gaussian noise" (the authors: J. Witulska, A. Zaleska, N. Kremzer-Osiadacz, A. Wyłomańska, I. Jabłoński).


The provided functions allow us to detect single change in scale for two-regimes data. The change point detection can be done using robust versions of two known methods:
- ICSS (reference: Inclan, C., & Tiao, G. C. (1994). Use of cumulative sums of squares for retrospective detection of changes of variance. Journal of the American Statistical Association, 89(427), 913-923),
- quantile method named here OLS (reference: G. Janusz, G. Sikora, A. Wyłomańska,  Regime variance testing - a quantile approach, Acta Physica Polonica B (2013)


Proposed approach is better than baseline methods for heavy-tailed data (especially with infinite variance) and comparably good for Gaussian data.
