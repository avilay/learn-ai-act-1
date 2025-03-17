# Basic Chart Types
### Line
This is needed with all three x-value types.

### Bar
The main requirement is to draw vertical bars. Horizontal bars are nice to have. This is needed with all three x-value types.

### Graph
This is more like the OS-X Grapher application, where I input a function and the range, the system draws a line graph. This only works with numerical x-values.

### Scatter
At its most basic this type of graph is able to represent 2-D data with numerical x-values. I also need a way to easly control other characteristics like the shape, size, and color of the marker. This can then be used to represent multiple dimensions of the dataset. E.g., I can use the size of the marker to represent a numerical third dimension, the shape of the marker can be a categorical fourth dimension, and the color can be the fifth dimension (say the target of an ML dataset).

### Histogram
This works on 1D dataset. It should be able to bin the dataset and draw the histogram.

### Images
Given an image read from Pillow, this should be able to draw the image on a plot - whether the image is RGB, grayscale, or RGBA.

# Chart Annotations
In addition to drawing the basic chart types, I also need to be able to do the following easily -

  1. [x] Add a title to the chart.
  2. [x] Add x-axis and y-axis labels.
  3. [x] Add a legend to the chart.
  4. [x] Have the labels auto-format based on available space.
  5. [x] Control the tick intervals.
  6. [x] Control the grid line visibility.
  7. [x] Control the color of the plots.
  8. [x] Draw multiple charts in the same figure.
  9. [x] Overlay charts on top of each other.
  10. [x] Have timeseries data auto group.
  11. [x] Draw matrix of scatter plots.
  12. [x] Draw hundreds of thousands of data points without too much of a lag.
  
  1. [x] Categorical Line
  2. [x] Numerical Line
  3. [x] Categorical Bar
  4. [x] Numerical Bar
  5. [x] Graph
  6. [x] Scatter with 2D
  7. [x] Scatter with 4D
  8. [x] Scatter matrix
  9. [x] Image
  10. [x] Timeseries line
  11. [x] Timeseries bar
  
# Scenarios

## Scenario 1
Have a dataset with a lot of rows that looks like:

```
+--------+--------+
| TOPIC  | DEMAND |
+--------+--------+
| Apple  |  20.   |
| Orange |  45.   |
| Pear   |  12.   |
+--------+--------+
```

Draw a vertical bar plot that should have a title, x-axis, and y-axis labels. The x-axis labels should autoformat so they show up properly in a small plot space.

## Scenario 2
Have a dataset with a lot of rows that looks like:

```
+--------+--------+--------+
| FRUITS | DEMAND | SUPPLY |
+--------+--------+--------+
| Apple  |  20.   |  27.   |
| Orange |  45.   |  30.   |
| Pear   |  12.   |  15.   |
+--------+--------+--------+
```

Draw two line plots on the same chart area - one for supply and one for demand. The chart should have a title, auto-formatted x-axis labels, y-axis labels. In addition the chart should also have a legend to indicate which line is demand and which one is supply.

## Scenario 3
Have a dataset with a lot of rows that looks like:

```
+-----------+------------+
| READ TIME | PAGE VIEWS |
+-----------+------------+
| 3         | 100        |
| 5         | 73         |
| 15        | 12         |
+-----------+------------+
```

Draw two different charts - one representing the data as a line plot and another as a bar plot in the same figure. Both the charts should have titles with x and y-axis labels. It should be possible to control the colors as well as colormaps for both the charts.

## Scenario 4
Have a dataset that looks like this:

```
+--------------+-------------+
| SEPAL LENGTH | SEPAL WIDTH |
+--------------+-------------+
| 12           | 24          |
| 5            | 8           |
| 13           | 18          |
+--------------+-------------+
```

Draw a scatter plot with the appropriate titles and legends. It should be possible to control the tick intervals that show up on both the x- and y-axes.

## Scenario 5
Have a dataset that looks like:

```
+------+-----+------+-----+--------+
| BILL | TIP | SIZE | SEX | SMOKER |
+------+-----+------+-----+--------+
|10.50 |1.25 | 2.   | M.  | Y      |
|10.50 |1.25 | 2.   | M.  | Y      |
|10.50 |1.25 | 2.   | M.  | Y      |
+------+-----+------+-----+--------+
```
Draw a scatter plot with BILL on the x-axis, TIP on the y-axis. The color of the marker should depend on the sex of the tipper. Its size should depend on the size of the dinner party. And its shape should depend on whether there was a smoker in the group.

## Scenario 6
Have a dataset that has 4 numerical attributes. Draw 12 scatter plots that plot each attribute against the other.

## Scenario 7
Get a timeseries dataset with 100K rows and day or even smaller granular rows. Try to plot them all at once with a line graph and then with a bar graph. Does the plotting library provid a way to autogroup the data?

## Scenario 8
Draw a continuous graph similar to the Grapher Mac OSX app for $e^x$, $log_2(x)$, $0.5^x$. Toggle grid lines off and on. Draw the graphs on different charts in the same figure. Draw them in the same chart overlaid on top of each other with a legend.

## Scenario 9
Get a dataset of 10 or so smallish grayscale images. Plot them side-by-side with their labels. Do the same exercise with RGB(A) images.

## Scenario 10
Historgrams and distribution plots.

# Exploration
Quick impressions of the various charting libraries I have tried and failed to like.

### Matplotlib
Don't like this because the API is pretty convoluted and very hard to use. Even after I wrote plotter, it was too hard to use even for my simple glucose history chart.

### Bokeh
Don't like this because the API documentation is pretty bad. It was not easy to figure out what the different function arguments expect. Also, it seems too tightly integrated with pandas. Giving it another shot because the timeseries functionality is really good. Besides it also does images. Maybe its worth my time to (re)learn pandas.

### Plotly
Requires an API key and username. This means that sooner or later they'll start asking for money.

#### Update
Contrary to my prediction above, they actually open sourced their charting library and decoupled it from their main revenue stream of hosting charts. I am giving another shot (May 2020).

### PyGal
Exploring now. Does not support jupyter OOB it seems. Timeseries functionality is not as good as bokeh. API documentation seems just as dense.
