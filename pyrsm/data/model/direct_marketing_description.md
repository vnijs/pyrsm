## MGT 403: Direct Mail Exercise
 
In this data set, you will find data for 1000 customers of a direct mail catalog company who just placed an order. For each customer, there is information on:
 
- AmountSpent: the amount of money spent by the customer in dollars
- Age (Old, Middle, Young): the customer's age
- Gender (Male, Female): the customer's gender
- OwnHome (Own, Rent): whether the customer owns a home 
- Married (Single, Married): whether the customer is married
- Location (Far, Close): the distance from the customer's address to the nearest brick and mortar store selling comparable products
- Salary: annual salary of the customer in dollars
- Children (0, 1, 2, 3): number of children the customer has
- History (High, Medium, Low, None): the customer's previous purchase volume, None means the customer has never purchased before
- Catalogs: number of catalogs sent to the customer
 
#### Questions about the _View_ tab:
1. Use the _View_ tab to determine how many old males live close to a brick and mortar store (click the ? icon on the bottom left of your screen for documentation).
2. How many people own their own home but have a salary below $15K?
3. Are there single women with 2 or 3 kids? If so, how many? What does that suggest about the definition of the Single variable?
 
#### Questions about the _Explore_ and _Visualize_ tabs:
1. Use the _Explore_ tab to generate basic descriptive statistics for the variable AmountSpent.
2. Use the `Group by` filter to group these results by Age. For `Apply functions`, choose 25%, 75%, Max, Min, and Median.
3. Use the _Visualize_ tab to create a box-plot with Age as the X-variable and AmountSpent as the Y-variable. Compare the output to results you obtained in the _Explore_ tab. Try to describe what the box plots illustrate.
4. Use the _Visualize_ tab to determine the age group with the most consumers
 
#### On Box Plots:
1. rnd1 and rnd2 are randomly generated variables. One of the two variables was drawn from a uniform distribution and the other variable was drawn from a normal distribution. Which one is which? Try creating a histogram.
2. Generate a box plot of rnd1 by Age group. If you compare this box-plot to the box-plot of Age versus AmountSpent you made previously, why is the whisker for AmountSpent so much shorter  at the bottom? Take a look at the help file (? icon on the bottom left) to better understand what is shown in the plot. What is different about the distribution of rnd1 and AmountSpent? 
3. Create a histogram with AmountSpent as the X-variable and Age as the Facet col variable. What does the Facet col selection do?
4. Looking at the plot in the last step, if you were to make box plots of the same, could you have predicted which should have the shortest bottom whisker?
 
#### Visualization Challenge
1. Show the relationship between AmountSpent and Salary in a graph
2. Are there differences in AmountSpent and Salary for married and single customers?
3. Can you show all this information in one single plot (i.e., without Facet row or Facet column)?
 
Click for more information about <a href="http://rmarkdown.rstudio.com/authoring_pandoc_markdown.html" target="_blank">markdown</a>