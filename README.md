# Market-trends
Combining popular trading techniques with well-tailored indicators produces sufficient buy-sell signal features of which the machine learning weighs respectively for making a high threshold final decision to buy or sell on a particular day. Including micro and macro time scale series is critical for determining the cumulative trend signal for various technical indicators and price derivatives. The machine learning model aggregates all signals across various time series (i.e. hourly, daily, weekly, and monthly data points) and returns a percent probability of success.

The motivation of this research is to find better fit ways to classify buying and selling opportunities in the stock market. The problem with this task is that making such predictions often results in near a 50% success rate with basic public models and near 60-70% for professional models where the longer the time frame the more accurate the prediction. Therefore, the topmost used indicators and their best use strategies will be a way to subclassify the data into critical features of interest on the decision to buy or sell. My contribution will apply to making better generalized predictions framed within each month based on moving and momentum averages of the stock price. 

This research partitions X labels of the data into a daily dataset for interpreting micro and macro buy-sell signals using top-level indicators, making up a total of 36 features. The 3-class Y label, or ground truth, was created by finding the 25th percentile of the high and low of the stock price in a given month. The stock price on any given day within the top 25% of its 1 month high represents a sell signal (-1.0), within the bottom 25% of its 1 month low represents a buy signal (1.0), and the remaining 50% in between represents a hold signal (0.0). 

The final results of this study use data from Apple stock. The linear regression model resulted in 75% accuracy in a 1-year time frame and 71% accuracy in a 10-year time frame.

### v1.1.1 Alpha Release
This release needs much tidying after Yahoo Finance changed their availability of data since late-2021.
The resulting errors are due to a change in time variables that need to be accounted for and normalized once again.
