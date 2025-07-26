1) /company/{company_id}/metrics
Returns a list of all available metrics (columns) for the company.
Use this to let the frontend choose which metrics to plot.

2) /company/{company_id}/trend/{metric}
Returns the time series data for a single metric (e.g., Revenue, EBITDA, volatility, etc.) for the given company.
Output: list of {period, value} pairs.

3) /company/{company_id}/trend/multi?metrics=Revenue&metrics=NetIncome
Returns time series for multiple metrics at once (for multi-line or comparison graphs).
Output: dict of metric name to list of {period, value} pairs.

4) Moving Average
Endpoint: /company/{company_id}/analytics/moving_average/{metric}?window=3
Description: Returns the moving average for a metric (default window=3).
Use: Smooths out short-term fluctuations to show longer-term trends.

4) Seasonality Decomposition
Endpoint: /company/{company_id}/analytics/seasonality/{metric}?period=4
Description: Returns trend, seasonal, and residual components using seasonal decomposition (default period=4, i.e., quarterly).
Use: Separates the time series into trend, seasonality, and noise for deeper insight.

5) Anomaly Detection
Endpoint: /company/{company_id}/analytics/anomaly/{metric}?z_thresh=2.5
Description: Returns periods where the metric is an outlier (z-score above threshold, default 2.5).
Use: Identifies unusual spikes or drops for alerting or investigation.