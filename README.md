# Market-trends
Machine learning predictor for buying and selling opportunities using micro and macro market trends.
Micro and macro time scale references are critical for determining the cumulative trend signal for various technical indicators and price derivatives.

The machine learning model aggregates all signals across various time scales (i.e. hourly, daily, weekly, and monthly data points) and returns a percent probability of success. A buy and sell percent threshold must be initially defined and updated to reflect conservative action.

### v1.1.1 Alpha Release
This release needs much tidying after Yahoo Finance changed their availability of data since late-2021.
The resulting errors are due to a change in time variables that need to be accounted for and normalized once again.
