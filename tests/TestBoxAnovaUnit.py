import unittest
from BoxAnova import BoxAnova, multiple_box_anova
import pandas as pd
import numpy as np


class TestBoxAnova(unittest.TestCase):
    def setUp(self):
        N = 1000
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
                    loc=(group_distributions[group]["mean"] + hue_distributions[hue]["mean"]) * 1000,
                    scale=(group_distributions[group]["stddev"] + hue_distributions[hue]["stddev"]) * 1000,
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
        self.df = pd.DataFrame(data, columns=["group", "hue", "value1", "value2", "value3", "value4"])
        self.box_anova = BoxAnova(df=self.df,variable="value1", group="group")

    def test_init(self):
        pd.testing.assert_frame_equal(self.box_anova.df, self.df)
        self.assertEqual(self.box_anova.variable, 'value1')
        self.assertEqual(self.box_anova.group, 'group')
        self.assertEqual(self.box_anova.order, ['Group1', 'Group2', 'Group3'])
        self.assertEqual(self.box_anova.title,  "Anova of 'Value1' by Group")

    def test_invalid_alpha_value(self):
        with self.assertRaises(ValueError, msg="alpha must be in [0.001, 0.01, 0.05, 0.1]"):
            BoxAnova(df=self.df, variable="value", group="group", alpha=0.2)

    def test_invalid_group_value(self):
        with self.assertRaises(ValueError, msg="Invalid not in columns"):
            BoxAnova(df=self.df, variable="value1", group="Invalid")

    def test_invalid_display_value(self):
        with self.assertRaises(ValueError, msg="display must be either 'group', 'hue' or 'both'"):
            self.box_anova.generate_box_plot(display='invalid')

    def test_missing_hue_for_display(self):
        with self.assertRaises(ValueError, msg="Hue must be provided if display is 'hue' or 'both'"):
            self.box_anova.generate_box_plot(display='hue')

    def test_invalid_method_value(self):
        with self.assertRaises(ValueError, msg="Method must be either bonf or sidak"):
            BoxAnova(df=self.df, variable="value", group="group", method="invalid")

    def test_save_no_fig(self):
        # Test that save method works without raising an exception
        with self.assertRaises(ValueError, msg="No figure to save"):
            self.box_anova.save(picture_path='.', file_prefix='test', file_suffix='suffix', create_folder=False)

    def test_plot_box_plot(self):
        # Test that plot_box_plot method works without raising an exception
        try:
            self.box_anova.plot_box_plot()
        except Exception as e:
            self.fail(f"plot_box_plot method raised exception {e}")

    def test_generate_box_plot(self):
        # Test that generate_box_plot method works without raising an exception
        try:
            self.box_anova.generate_box_plot(display='group')
        except Exception as e:
            self.fail(f"generate_box_plot method raised exception {e}")

    def test_multi_group(self):
        multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=self.df, group="group",
                           display='group')

    def test_multi_group_no_fliers(self):
        multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=self.df, group="group",
                           display='group', box_kws={"showfliers": False})

    def test_multi_group_hue(self):
        multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=self.df, group="group",
                           hue="hue")#

    def test_multi_group_hue_no_fliers(self):
        multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=self.df, group="group",
                           hue="hue", box_kws={"showfliers": False})

    def test_multi_group_hue_reserved(self):
        multiple_box_anova(variables=["value1", "value2", "value3", "value4"], data=self.df, group="hue",
                           hue="group")

    # def invalid_order_value(self):
    #     with self.assertRaises(ValueError, msg="Invalid order not in list of groups"):
    #         BoxAnova(df=self.df, variable="value", group="group", order=["Invalid"])
    #
    # def invalid_orient_value(self):
    #     with self.assertRaises(ValueError, msg="Orient must be either 'v' or 'h'"):
    #         BoxAnova(df=self.df, variable="value", group="group", orient="invalid")
    #
    # def invalid_palette_value(self):
    #     with self.assertRaises(ValueError, msg="Invalid palette"):
    #         BoxAnova(df=self.df, variable="value", group="group", palette="invalid")
    #
    # def invalid_background_color_value(self):
    #     with self.assertRaises(ValueError, msg="Invalid background color"):
    #         BoxAnova(df=self.df, variable="value", group="group", background_color="invalid")
    #
    # def invalid_variable_value(self):
    #     with self.assertRaises(ValueError, msg="Invalid variable not in columns"):
    #         BoxAnova(df=self.df, variable="Invalid", group="group")
    #
    # def invalid_save_settings_value(self):
    #     with self.assertRaises(ValueError, msg="Invalid save settings"):
    #         self.box_anova.save(picture_path='.', file_prefix='test', file_suffix='suffix', dpi='invalid', pic_type='invalid', create_folder=False)
    #
    # def invalid_multiple_box_anova_value(self):
    #     with self.assertRaises(ValueError, msg="Invalid variables not in columns"):
    #         multiple_box_anova(variables=["Invalid"], data=self.df, group="group")
    #
    # def invalid_multiple_box_anova_display_value(self):
    #     with self.assertRaises(ValueError, msg="display must be either 'group', 'hue' or 'both'"):
    #         multiple_box_anova(variables=["value"], data=self.df, group="group", display='invalid')
    #
    # def invalid_multiple_box_anova_orient_value(self):
    #     with self.assertRaises(ValueError, msg="Orient must be either 'v' or 'h'"):
    #         multiple_box_anova(variables=["value"], data=self.df, group="group", orient='invalid')


if __name__ == '__main__':
    unittest.main()
