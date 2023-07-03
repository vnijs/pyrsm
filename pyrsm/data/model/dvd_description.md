## DVD

### Description

The data contain information on a sample of 20,000 customers who received an "instant coupon." The value of the coupon was varied between \$1 and \$5 and randomly assigned to the selected customers.

Our interest is in estimating the effect of the coupon on purchase of a newly released DVD. We will also investigate the role of two additional variables: `purch` is a measure of frequency of purchase and `last` is a measure of the recency of the last purchase by a customer. These measures are often used in practice to predict response rates.

Customers who received the coupon and purchased the DVD are identified in the data by the variable `buy`. Because the variable we want to predict is binary (`buy` = `yes` if the customer purchased the DVD and `buy` = `no` if she did not), logistic regression is appropriate.

### Variables

* buy = `yes` if the customer purchased the DVD and `no` if she did not
* coupon = value of an "instant coupon" in dollars. the value varies between \$1 and \$5
* purch = number of purchases by the customer in the past year
* last = days since the last purchase by the customer
* training = 70/30 split, 1 for training sample, 0 for validation sample

### Source

The dataset `dvd.rds` is available for download from <a href = "https://radiant-rstats.github.io/docs/examples/dvd.rds" target="_blank">GitHub</a>.