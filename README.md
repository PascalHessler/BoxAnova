# BoxAnova

BoxAnova is a Python package built on top of Seaborn Boxplots. Its main purpose is to create Boxplots with additional significance information. By using ANOVAs to evaluate if group and hue differences are significant, it adds significance information to the plot.

## Installation

You can install BoxAnova using pip:

```bash
pip install BoxAnova
```

or Poetry
```bash
poetry add BoxAnova
```

## Usage

Here is a simple example of how to use BoxAnova:

```python
from BoxAnova import BoxAnova
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize BoxAnova with your DataFrame, the group column, and the value column
box_anova = BoxAnova(df, variable='Value2', group='group')

# Plot the box plot with mean differences
box_anova.generate_box_plot()
```
![img.png](img.png)
### Options
There are some arguments you can set to change how the plots are generated:
- You can change orientation
```python 
box_anova = BoxAnova(df, variable='Value2', group='group', orient="V")
box_anova.generate_box_plot() 
```
![img_3.png](img_3.png)
- If you want using stars or exact values
```python 
box_anova = BoxAnova(df, variable='Value2', group='group', show_p_value=True)
box_anova.generate_box_plot() 
```
![img_1.png](img_1.png)
- If N of the groups should be displayed   
```python box_anova.generate_box_plot(fine_tuning_kws={"show_n": True}) ```
![img_2.png](img_2.png)


## Hue Argument
Finally, you can add a second group.
```python
box_anova.generate_box_plot(display='hue', hue="hue")
```
![img_5.png](img_5.png)
# Muli Box Anova
The main idea is you might want to genreate multiple plots at once. MultipleBOxAnova supports all functions of BoxAnova in addition it allows for multiple Variables. 
For multiple box anova, you can use the `multiple_box_anova` function:

```python
from BoxAnova import multiple_box_anova
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Call the multiple_box_anova function
multiple_box_anova(variables=["first_variable", "second_variable"], data=df, group="group")
```

In both versions you can add the hue argument.

```python
from BoxAnova import multiple_box_anova, BoxAnova
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# BoxAnova
# Initialize BoxAnova with your DataFrame, the group column, and the value column
box_anova = BoxAnova(df, variable='first_variable', group='group_column' )

# Plot the box plot with hue
box_anova.plot_box_plot(hue="hue")

# Call the multiple_box_anova function
multiple_box_anova(variables=["first_variable", "second_variable"], data=df, group="group_column", hue="hue")

```
The multiple_box_anova does not require the initialization of the BoxAnova class

# Experiemntal
- Striplot in Boxplot BoxAnova(stripplot=True)
- Violinplot alternativ zu Boxplot BoxAnova(violin=True)

# Contributing

Contributions are welcome!

