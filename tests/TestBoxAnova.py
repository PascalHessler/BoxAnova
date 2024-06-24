import pandas as pd
import numpy as np
from BoxAnova import BoxAnova, multiple_box_anova
import pytest

@pytest.fixture
def generate_sample_data(N: int = 1000) -> pd.DataFrame:
    groups = ["Group1", "Group2", "Group3"]
    hues = ["Hue1", "Hue2"]

    # Definieren Sie die Parameter für die Normalverteilungen für jede Gruppe
    group_distributions = {
        "Group1": {"mean": 30, "stddev": 8},
        "Group2": {"mean": 10, "stddev": 8},
        "Group3": {"mean": 50, "stddev": 8},
    }
    hue_distributions = {
        "Hue1": {"mean": 5, "stddev": 10},
        "Hue2": {"mean": 0, "stddev": 0},
    }
    # Erstellen Sie die Daten
    data = []
    for group in groups:
        for hue in hues:
            group_data = np.random.normal(
                loc=(group_distributions[group]["mean"] + hue_distributions[hue]["mean"])*1000,
                scale=(group_distributions[group]["stddev"] + hue_distributions[hue]["stddev"])*1000,
                size=N,
            )
            group_data2 = np.random.beta(
                a=group_distributions[group]["mean"] + hue_distributions[hue]["mean"],
                b=group_distributions[group]["stddev"] + hue_distributions[hue]["stddev"],
                size=N,
            )
            group_data3 = np.random.lognormal(
                mean=group_distributions[group]["mean"] + hue_distributions[hue]["mean"],
                sigma=group_distributions[group]["stddev"] + hue_distributions[hue]["stddev"],
                size=N,
            )
            group_data_4 = np.random.normal(
                loc=(group_distributions[group]["mean"] + hue_distributions[hue]["mean"]),
                scale=(group_distributions[group]["stddev"] + hue_distributions[hue]["stddev"]),
                size=N,
            )
            data.extend(zip([group] * N, [hue] * N, group_data, group_data2, group_data3, group_data_4))

    # Erstellen Sie einen DataFrame aus den Daten
    return pd.DataFrame(data, columns=["group", "hue", "value1", "value2", "value3"])


def test_group_and_hue(generate_sample_data):
    df = generate_sample_data
    box = BoxAnova(df=df, variable="value1", group="group", orient="h")
    box.generate_box_plot(display="both", show=True, hue="hue")


def test_group():
    df = generate_sample_data()
    box = BoxAnova(df=df, variable="value1", group="group", orient="h")
    box.generate_box_plot(display="group", show=True)


@pytest.fixture
def box_anova():
    df = pd.DataFrame({
        'group': ['Group1', 'Group2', 'Group1', 'Group2'],
        'value': [1, 2, 3, 4]
    })
    return BoxAnova(df=df, variable='value', group='group')

def test_init(box_anova):
    assert box_anova.df.equals(df)
    assert box_anova.variable == 'value'
    assert box_anova.group == 'group'

def test_save(box_anova):
    # Test that save method works without raising an exception
    try:
        box_anova.save(picture_path='.', file_prefix='test', file_suffix='suffix', create_folder=False)
    except Exception as e:
        pytest.fail(f"Save method raised exception {e}")

def test_check_and_init(box_anova):
    # Test that check_and_init method works without raising an exception
    try:
        box_anova.check_and_init()
    except Exception as e:
        pytest.fail(f"check_and_init method raised exception {e}")

def test_plot_box_plot(box_anova):
    # Test that plot_box_plot method works without raising an exception
    try:
        box_anova.plot_box_plot()
    except Exception as e:
        pytest.fail(f"plot_box_plot method raised exception {e}")

def test_calc_sig(box_anova):
    # Test that calc_sig method works without raising an exception
    try:
        box_anova._calc_sig(group_order=['Group1', 'Group2'])
    except Exception as e:
        pytest.fail(f"calc_sig method raised exception {e}")

def test_generate_box_plot(box_anova):
    # Test that generate_box_plot method works without raising an exception
    try:
        box_anova.generate_box_plot(display='group')
    except Exception as e:
        pytest.fail(f"generate_box_plot method raised exception {e}")



def test_multi_group():
    df = generate_sample_data()
    multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=df, group="group")


def test_multi_group_hue():
    df = generate_sample_data()
    multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=df, group="group",
                       hue="hue")