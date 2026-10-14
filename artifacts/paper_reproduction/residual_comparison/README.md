# Residual comparison provenance

These artifacts use the original epoch-34 checkpoint and aligned Prophet
predictions, not the new strict 60/20/20 checkpoint. Rebuild using
`python reproduction/plot_residual_comparison.py` in the reproduction environment.

`residual_histogram_author_method.png` follows the author's plot_hist function
with 80 bins per model. Each value is the signed mean of 24 hourly residuals:
8,595 origins per model. `residuals_mean24h_per_origin.csv` and
`author_method_summary.json` retain the auditable data and metadata. Frequency
counts windows, not individual hourly forecast pairs. The full-range image
keeps the tails visible; the author-style horizontal zoom hides 169 LSTM and
165 Prophet values without removing them from evaluation.

`residual_histogram_local.png` is the earlier individual-pair plot (206,280
pairs), a different aggregation. Do not confuse it with the mean-24h histogram.
The prediction cache contains the checkpoint SHA-256 and is rejected if the
checkpoint changes. Both histograms preserve the actual computed counts.

`paper_fig3_original.jpg` is Figure 3 from Gürses-Tran, Oppermann and Monti,
ProLoaF: Probabilistic load forecasting for power systems, SoftwareX 23 (2023)
101487, page 6, DOI https://doi.org/10.1016/j.softx.2023.101487 (CC BY 4.0).
Figure credit is retained here; local charts are separately labeled artifacts.
The exact checkpoint/data used to generate the published figure are not
confirmed. Similar styling or aggregation does not establish numerical fidelity.
LSTM and Prophet also differ in input information and fitting protocol.
