REPRODUCIBILITY PACKAGE

All records are simulated. The experiments do not establish clinical efficacy or medical-device validity.

Run with Python 3.12:
  python -m pip install -r requirements.txt
  python neuro_fuzzy_study.py --output regenerated_results --zip

The program checks missing dependencies and prints an installation command. Alternatively:
  python neuro_fuzzy_study.py --install-missing --output regenerated_results --zip

Use the default seeds and bootstrap count to reproduce the manuscript: data seed 20261025; split seeds 11, 29 and 47; 1,000 bootstrap resamples. The primary results use split seed 11. Floating-point libraries and hardware can cause small numerical and timing differences. PDF metadata and measured timing are not expected to be byte-identical across runs.

5.1 contains classification data, predictions, Tables 1-2 and Figures 1-2.
5.2 contains monitoring and sensor data, predictions, Tables 3-4 and Figures 3-4.
5.3 contains measurement stress, Tables 5-6, Figures 5-6, fitted neuro-fuzzy parameters, rule traces, paired comparisons, repeated-split and subgroup metrics, verified reference metadata, and the execution manifest.

CSV values are unrounded. When recalculating rank-based metrics with pandas, use pd.read_csv(path, float_precision="round_trip") to preserve floating-point ties exactly. Figures are supplied as high-resolution PNG and vector PDF. Synthetic datasets contain the primary partition labels. Predictions retain original row and subject identifiers. The pressure threshold is illustrative and is not a treatment recommendation. Intervals condition on the fitted models and do not include training-sample uncertainty.

VERIFICATION.txt records manuscript and numerical checks. SHA256.json contains hashes for every other file in this archive. The complete source code and the exact package versions used are included.
