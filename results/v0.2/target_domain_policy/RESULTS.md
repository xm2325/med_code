# MedCode v0.2.2 target-domain policy confirmation

Calibration AUTO coverage: 48.2%
Calibration empirical Accuracy@1: 96.7%
Calibration one-sided lower bound: 95.0%
Fresh confirmatory AUTO coverage: 48.5%
Fresh confirmatory Accuracy@1: 96.9%
95% release gate pass: True

The 95% AUTO policy passes only if the threshold is selected without confirmatory labels on fresh target-domain calibration data using the one-sided lower-bound rule, and the locked threshold then reaches at least 95% exact-code agreement on the fresh disjoint confirmatory sample.
