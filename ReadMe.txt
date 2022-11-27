Predictions for energy production through Machine Learning(ML) techniques

A data file is received (TrainSet.csv), containing information about:
> datetime_utc: Day and Time
> Generation_BE: Scheduled energy production in Belgium
> Generation_FR: Scheduled energy production in France
> Prices.BE: System Marginal Price 
> holidaysBE: National holidays for Belgium (1 for holidays, otherwise 0)

The program:
> Cleanses the data received (manages outliers & fills NA values)
> Makes hourly predictions for the next 7 days and saves them to the "TrainSet.csv" file through the most efficient ML technique.

ML techniques examined:
> Multiple linear regression
> Naïve Bayes
> Support Vector Machine
> Decision tree
> Random Forest
> Gradient boosting trees
> Neural Networks