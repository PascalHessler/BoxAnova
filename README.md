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
box_anova = BoxAnova(df, variable='first_variable', group='group' )

# Plot the box plot
box_anova.plot_box_plot()
```

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


## Contributing

Contributions are welcome!

