# News Mood

Some data science practice pinging Twitter's API and doing sentiment analysis.

1. Scatter Plot of the last 100 tweets sent out by BBC, CBS, CNN, Fox, and NYT, ranging from -1.0 to 1.0 based on sentiment.  0 is neutral, -1 is negative, and 1 is positive.  Each plot point will be the *compound* sentiment of the tweet.  Points will be sorted by timestamp.
2. Bar Plot visualizing the overall sentiments of the last 100 tweets from each organization.  Again, we will aggregate the *compound sentiments*.

We will need to use `tweepy, pandas, matplotlib, seaborn, textblob, and VADER`.

The final notebook must:

 - Pull the last 100 tweets from each outlet.
 - Perform a sentiment analysis with compound, positive, neutral, and negative scoring for each tweet.
 - Pull source account, text, date, compound, positive, neutral, and negative scores into a DataFrame.
 - Export the data into a CSV.
 - Save PNG's of each plot.
 - Observe trends.
 - Label plot axes and titles.  Plot titles must include date.

# Plot Bot

Creating a twitter bot that sends out visualized sentiment analysis using the Twitter API.

 - Bot scans account every 5 minutes for mentions.
 - Bot should pull 500 most recent tweets to analyze for each incoming request.
 - Bot should only analyze an account once per script run.
 - Plots should have legends and labels.
 - It should mention the name of the account that asked for analysis.
 - Deliverables include 3 actual analyses.