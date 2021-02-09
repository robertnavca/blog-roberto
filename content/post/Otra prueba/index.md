## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

![](index_files/figure-markdown_github/pressure-1.png)

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.

``` python
import pandas as pd
```

``` python
data = pd.read_csv("data/train.csv", index_col='Id')
```

``` python
data.head()
```

    ##     MSSubClass MSZoning  LotFrontage  ...  SaleType SaleCondition SalePrice
    ## Id                                    ...                                  
    ## 1           60       RL         65.0  ...        WD        Normal    208500
    ## 2           20       RL         80.0  ...        WD        Normal    181500
    ## 3           60       RL         68.0  ...        WD        Normal    223500
    ## 4           70       RL         60.0  ...        WD       Abnorml    140000
    ## 5           60       RL         84.0  ...        WD        Normal    250000
    ## 
    ## [5 rows x 80 columns]
