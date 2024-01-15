# Logistic Regression on Parallel Time Frames of Financial Market Indicators
Combining popular trading techniques with well-tailored indicators produces sufficient buy-sell signal features of which logistic regression weighs respectively for making a high threshold decision to hold, buy, or sell on a particular day. Including micro and macro time scale series is critical for determining the cumulative trend signal for various technical indicators and price derivatives. The supervised machine learning model aggregates all signals across parallel time frames (i.e. hourly, daily, weekly, and monthly data points) and returns a percent probability of success.

The motivation of this research is to find better fit ways to classify buying and selling opportunities in financial markets. The problem with this task is that making such predictions often results in near a 50% success rate with basic public models and near 60-70% for professional models where the longer the time frame the more accurate the prediction. Therefore, the topmost used indicators and their best use strategies will be a way to subclassify the data into critical features of interest on the decision to buy or sell. My contribution will apply to making better generalized predictions framed within each month based on moving and momentum averages of the asset's price. 

This research partitions X labels of the data into a daily dataset for interpreting micro and macro buy-sell signals using top-level indicators, making up a total of 36 features. The 3-class Y label, or ground truth, was created by finding the 25th percentile of the high and low of the stock price in a given month. The stock price on any given day within the top 25% of its 1 month high represents a sell signal (-1.0), within the bottom 25% of its 1 month low represents a buy signal (1.0), and the remaining 50% in between represents a hold signal (0.0). 

The final results of this study use data from Apple stock. The linear regression model resulted in 75% accuracy in a 1-year frame and 71% accuracy in a 10-year frame.

### v1.0.9 Beta Release
Adding macro time intervals within a 10-year frame. Time intervals (i.e. daily, weekly, and monthly) represent the X-axis of each price data point on the Y-axis. Training is done on data from 2011 to 2021 for Apple, Telsa, and Bitcoin.

The final scoring comes out to 70.7% accuracy but with a tendency to sell rather than buy. Ideally, buying to hold would result in the greatest returns. An ideal selling point would occur when a "Strong Sell" signal arises. This did not appear, because evidently the price action of the following tickers remained in a macro bull-run. Future experiments can use new data during bear markets to better train and test the machine learning model algorithm.

### v1.1.1 Alpha Release
This release is for non-production research purposes. Yahoo Finance changed their format and availability of data since late-2021. The resulting errors are due to a change in time variables that need to be accounted for and normalized once again.
