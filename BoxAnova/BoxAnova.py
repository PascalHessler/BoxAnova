import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import os
from scipy.stats import f_oneway, ttest_ind
import statsmodels.stats.multicomp as mc
import math
from typing import Literal, TypedDict
import warnings
from BoxAnova.utils.renaming import formatting, renaming


class SaveSettings(TypedDict, total=False):
    picture_path: str
    file_prefix: str
    file_suffix: str
    dpi: int
    pic_type: str
    create_folder: bool


class Labels(TypedDict, total=True):
    labels: dict
    seperator: str


def multiple_box_anova(variables: list, data: pd.DataFrame, group: str, hue: str = None, hue_order: list[str] = None,
                       display: Literal['group', 'hue', 'both'] = 'both', all_separate: bool = False,
                       orient: Literal["v", "h"] = "h",
                       titles: list | str = "",
                       subtitles: list | str = "",
                       notes: list | str = "",
                       additional_texts: list | str = "",
                       save_to_file: bool = False,
                       settings_save: SaveSettings = None,
                       get_labels: bool = False,
                       labels: Labels = None,
                       show_fig: bool = True,
                       box_kws: dict = None,
                       **kwargs):
    """
    Generating BoxPlots for multiple variables
    :param variables: For each variable a boxplot will be generated
    :param data: the DataFrame containing the data
    :param group: The Grouping variable, which will be used to separate the data
    :param hue:  The hue variable, which will be used to separate the data
    :param hue_order:  The order of the hue variable
    :param display: There are three types to display the data: 'group', 'hue', 'both'. With 'group' the data will be
    only split by grop and the Anova is only based on this.
    When using "hue" both group and hue are used to plot the boxplot, but the Anova is only shown or the differences in hue by group.
    With 'both' the data is split by group and hue and the Anova is based on the group and hue separately.
    :param all_separate: This allows to create plots for all three display types.
    :param orient: You can choose between 'v' and 'h' for vertical and horizontal orientation
    :param titles: The titles of the plots, when no title is provided the variable name will be used like this 'Anova of ' + variable + ' by ' + group
    :param subtitles: The subtitles of the plots
    :param notes: The notes of the plots, these are additional information which will be displayed at the bottom of the plot
    :param additional_texts: Additional texts which will be displayed at the bottom of the plot
    :param save_to_file: If True the plots will be saved to a file, but the path must be provided in settings_save
    :param settings_save: The settings for saving the plots
    :param get_labels: TBA
    :param labels: TBA
    :param show_fig: If True the plots will be shown
    :param box_kws: Additional arguments for the boxplot passed to seaborn.boxplot
    :param kwargs: Additional arguments for the BoxAnova class
    :return:
    """
    if labels:
        titles = []
        subtitles = []
        for i in variables:
            if i not in labels["labels"]:
                raise AttributeError(f"The label Dict did not include all necessary variables: {i}")
            title, subtitle = labels["labels"][i].split(labels["seperator"])
            titles.append(title.strip())
            subtitles.append(subtitle.strip())

    if isinstance(titles, str):
        titles = [titles for _ in range(len(variables))]
    if isinstance(subtitles, str | None):
        subtitles = [subtitles for _ in range(len(variables))]
    if isinstance(notes, str):
        notes = [notes for _ in range(len(variables))]
    if isinstance(additional_texts, str):
        additional_texts = [additional_texts for _ in range(len(variables))]

    if not hue and group != "group":
        display: Literal['group', 'hue', 'both'] = "group"
        warnings.warn("Hue is not provided. Display type was changed to 'group'")

    for title, subtitle, note, additional_texts, var in zip(titles, subtitles, notes, additional_texts, variables):
        box = BoxAnova(df=data, variable=var, group=group, title=title, subtitle=subtitle, note=note,
                       additional_text=additional_texts, orient=orient, box_kws=box_kws, **kwargs)
        if all_separate:
            file_prefix = settings_save.get('file_prefix', "")
            settings_save["file_prefix"] = f"group_{file_prefix}"
            box.generate_box_plot(hue=hue, hue_order=hue_order, display="group", save=save_to_file, show=show_fig,
                                  settings_save=settings_save, )
            settings_save["file_prefix"] = f"hue_{file_prefix}"
            box.generate_box_plot(hue=hue, hue_order=hue_order, display="hue", save=save_to_file, show=show_fig,
                                  settings_save=settings_save)
            settings_save["file_prefix"] = f"both_{file_prefix}"
            box.generate_box_plot(hue=hue, hue_order=hue_order, display="both", save=save_to_file, show=show_fig,
                                  settings_save=settings_save)
        else:
            box.generate_box_plot(hue=hue, hue_order=hue_order, display=display, save=save_to_file, show=show_fig,
                                  settings_save=settings_save)


class BoxAnova:
    alpha_boarders = [0.001, 0.01, 0.05, 0.1]

    def __init__(self,
                 df: pd.DataFrame, variable: str,
                 group: str,
                 order: list[str] = None,
                 orient: Literal["v", "h"] = "h",
                 method: Literal["bonf", "sidak"] = "bonf",
                 alpha: float = 0.1,
                 use_corrected_p: bool = True,

                 palette=sns.color_palette('colorblind'),
                 background_color: str = "white",

                 title: str = "", subtitle: str = "",
                 note: str = "", additional_text: str = "",

                 show_fig: bool = True,
                 box_kws: dict = None,
                 **kwargs):
        """

        :param df: The DataFrame containing the data
        :param variable: The variable which should be used for the Anova
        :param group: The Grouping variable, which will be used to separate the data
        :param order: The order of the groups
        :param orient: The orientation of the plot 'v' for vertical and 'h' for horizontal
        :param method: The method for the correction of the p-values, either 'bonf' or 'sidak'
        :param alpha: The alpha value for the p-values
        :param use_corrected_p: If True the corrected p-values will be used, this is an correction when using Anova for multiple groups.
        :param palette: The color palette for the plot
        :param background_color: The background color of the plot
        :param title: The title of the plot
        :param subtitle: The subtitle of the plot
        :param note: The note of the plot
        :param additional_text: Additional text which will be displayed at the bottom of the plot
        :param show_fig: If True the plot will be shown
        :param box_kws: Additional arguments for the boxplot passed to seaborn.boxplot
        :param kwargs: Additional arguments for the BoxAnova class
        """
        self.df = df
        self.variable = variable
        self.orient = orient
        self.group = group

        self.alpha = alpha

        self.title = title
        self.subtitle = subtitle
        self.note = note
        # adds a linebreak when additional text is not empty
        self.additional_text = "\n" + additional_text if additional_text else additional_text

        self.method = method

        self.check_and_init()

        if order is None:
            order = list(self.df[group].unique())
        self.order = order

        self.use_corrected_p = use_corrected_p

        self.background_color = background_color
        self.palette = palette

        self.fig = None
        self.ax = None
        self.show_fig = show_fig
        self.kwargs = kwargs

        if box_kws is None:
            box_kws = {}
        self.box_kws = {"showmeans": True, "meanprops": {"markerfacecolor": "white", "markeredgecolor": "black"},
                        **box_kws}

    @staticmethod
    def show():
        plt.show()

    def save(self, picture_path: str = "",
             file_prefix: str = "",
             file_suffix: str = "",
             dpi=300,
             pic_type: str = "png",
             full_path: callable = None,
             create_folder: bool = False,
             **kwargs
             ):
        if self.fig is None:
            raise ValueError("No figure to save")
        if not os.path.isdir(picture_path):
            if create_folder:
                os.makedirs(picture_path, exist_ok=True)
            else:
                raise OSError("Directory does not exist")
        if full_path:
            full_path = full_path(self)
        else:
            full_path = os.path.join(picture_path,
                                     f"{file_prefix}{self.group}_{self.variable}{file_suffix}.{pic_type}")
        self.fig.savefig(full_path, dpi=dpi, bbox_inches='tight', **kwargs)
        plt.close()

    def check_and_init(self):
        if self.alpha not in self.alpha_boarders:
            raise ValueError(f"alpha must be in {self.alpha_boarders}")

        if self.group not in self.df.columns:
            raise ValueError(f"{self.group} not in columns")

        if not self.title:
            self.title = f"Anova of '{formatting(self.variable)}' by {formatting(self.group)}"

        if self.method not in ["bonf", "sidak"]:
            raise ValueError("Method must be either bonf or sidak")

    @property
    def start_point(self):
        return self.alpha_boarders.index(self.alpha)

    @property
    def max_value_on_scale(self):
        if self.box_kws.get("showfliers", True) is False:
            return self.df[self.variable].quantile(0.975)
        return self.df[self.variable].max()

    def plot_box_plot(self, hue: str = None, hue_order: list[str] = None, formatting_text: bool = True,
                      position_title: float = 1.04, position_offset: float = 0.05
                      ):
        # make sure canvas is clear
        plt.close()
        # prep arguments
        optional_params = {}
        if hue:
            optional_params['hue'] = hue
            optional_params['hue_order'] = hue_order

        # create plot
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor(self.background_color)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            if self.orient == "h":
                x = self.variable
                y = self.group
            else:
                x = self.group
                y = self.variable

            self.ax = sns.boxplot(data=self.df, x=x, y=y, order=self.order, ax=self.ax, palette=self.palette,
                                  **optional_params, **self.box_kws
                                  )
        if formatting_text:
            renaming(self.ax)

        # calculation position
        position_height = position_title
        if self.subtitle:
            position_height += position_offset
        if hue:
            position_height += position_offset
            sns.move_legend(self.ax, "lower center",
                            bbox_to_anchor=(.5,
                                            position_height - position_offset * 3 if self.subtitle else
                                            position_height - position_offset * 2),
                            ncol=len(self.df[hue].unique()), title=None, frameon=False, )

        self.ax.text(x=0.5, y=position_height, s=self.title, fontsize=14, weight='bold', ha='center', va='bottom',
                     transform=self.ax.transAxes)
        if self.subtitle:
            self.ax.text(x=0.5, y=position_height - position_offset, s=self.subtitle, fontsize=12, alpha=0.8,
                         ha='center', va='bottom', transform=self.ax.transAxes)

    def _calc_sig(self, group_order: list, annotation=True) -> tuple[any, pd.DataFrame]:
        gb = self.df.groupby("temp_group", observed=True)
        result_oneway = f_oneway(*[gb.get_group(group_)[self.variable] for group_ in gb.groups])
        if annotation:
            self.ax.annotate(self.annotation_text(result_oneway[1]),
                             xy=(1.0, -0.2), xycoords='axes fraction', ha='right',
                             va="center", fontsize=9)
        comp = mc.MultiComparison(self.df[self.variable], self.df["temp_group"], group_order=group_order)

        # Post hoc
        match self.method:
            case "bonf":
                post_hoc_res, a1, a2 = comp.allpairtest(ttest_ind, method="bonf", alpha=self.alpha)
            case "sidak":
                post_hoc_res, a1, a2 = comp.allpairtest(ttest_ind, method="sidak", alpha=self.alpha)
            case _:
                raise ValueError("Method must be either bonf or sidak")

        # wird zwar immer ausgerechnet aber nur für die Mittelwerte verwendet!
        mean_dif = comp.tukeyhsd(alpha=self.alpha)

        df_res = pd.DataFrame(post_hoc_res.data)
        df_res.columns = ["group1", "group2", "stat", "pval", "pval_corr", "reject"]
        df_res: pd.DataFrame = df_res[1:]

        return mean_dif, df_res

    def get_p_value(self, data: pd.Series) -> float:
        if self.method == "tukey":
            return float(str(data.values[3]))
        else:
            if self.use_corrected_p:
                return float(str(data.values[4]))
            else:
                return float(str(data.values[3]))

    def p_value_sig(self, p_value: float, mean_dif_single: float) -> str:
        # Überprüfen, ob p-Wert signifikant ist
        if p_value < self.alpha_boarders[self.start_point - 2]:
            text = f"{mean_dif_single:.2f}***"
        elif p_value < self.alpha_boarders[self.start_point - 1]:
            text = f"{mean_dif_single:.2f}**"
        elif p_value < self.alpha_boarders[self.start_point]:
            text = f"{mean_dif_single:.2f}*"
        else:
            text = f"{mean_dif_single:.2f}"
        return text

    def calc_tick_x_line(self, i, j, k, position_offset: float = 0, fix_position: bool = False) \
            -> tuple[float, float, float]:
        tick_size: float = math.ceil(self.max_value_on_scale * 0.1, ) - 0.5

        if fix_position:
            x_line = self.max_value_on_scale + 1 * tick_size
        else:
            x_line = self.max_value_on_scale + k * tick_size

        x_line += position_offset
        y_text = (j + i) / 2

        return tick_size, x_line, y_text

    def draw_lines(self, i, j, tick_size, x_line):
        # Draw bracket for values |_|
        if self.orient == "h":
            self.ax.plot([x_line, x_line], [i, j], lw=1, color='k')
            self.ax.plot([x_line - 0.1 * tick_size, x_line], [i, i], lw=1, color='k')
            self.ax.plot([x_line - 0.1 * tick_size, x_line], [j, j], lw=1, color='k')
        else:
            self.ax.plot([i, j], [x_line, x_line], lw=1, color='k')
            self.ax.plot([i, i], [x_line - 0.1 * tick_size, x_line], lw=1, color='k')
            self.ax.plot([j, j], [x_line - 0.1 * tick_size, x_line], lw=1, color='k')

    def draw_sig_level(self, sig_text, tick_size, x_line, y_text, **kwargs):
        settings_text = {"color": 'k', "fontsize": 10,
                         "horizontalalignment": "center", "verticalalignment": "center",
                         "rotation_mode": "anchor",
                         **kwargs}
        if self.orient == "h":
            plt.text(s=sig_text, x=x_line + tick_size / 1.8 - tick_size, y=y_text, rotation=90, **settings_text)
        else:
            plt.text(s=sig_text, x=y_text, y=x_line + tick_size / 1.8 - tick_size, rotation=0, **settings_text)

    def annotation_text(self, result_oneway: float) -> str:
        return (f"Anova p={result_oneway:.2f}, post-hoc {self.method}, α={self.alpha} \n Mean difference "
                f"*p < {self.alpha_boarders[self.start_point]}, "
                f"** p < {self.alpha_boarders[self.start_point - 1]}, "
                f"*** p < {self.alpha_boarders[self.start_point - 2]} \n "
                f"{self.note} {self.additional_text}")

    def generate_box_plot(self, hue: str = None, hue_order: list[str] = None,
                          display: Literal['group', 'hue', 'both'] = 'group', save=False, show=True,
                          settings_save: SaveSettings = None):

        if display not in ['group', 'hue', 'both']:
            raise ValueError("display must be either 'group', 'hue' or 'both'")
        if not hue and display in ["hue", "both"]:
            raise ValueError("Hue must be provided if display is 'hue' or 'both'")

        match display:
            case 'group':
                if hue:
                    warnings.warn("You provided the hue argument but selected to only show group."
                                  " This might be not intended")
                self.calc_sig_levels_group(hue=False)
            case 'hue':
                self.calc_sig_levels_hue(hue=hue, hue_order=hue_order, show_group=False)
            case 'both':
                self.calc_sig_levels_hue(hue=hue, hue_order=hue_order, show_group=True)
            case _:
                raise TypeError(f"Unexpected value for 'display' {display}. "
                                f"Only ['group', 'hue', 'both'] are valid options.")

        # solves issue with axis
        if self.orient == "h":
            self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        # clean up
        self.df.drop(columns="temp_group", inplace=True, errors="ignore")

        if show:
            self.show()
        if save:
            self.save(**settings_save)

        plt.close()

    def calc_sig_levels_group(self, hue=False, rate_position_offset_with_hue=0.1, fine_tuning_kws: dict = None):
        if fine_tuning_kws is None:
            fine_tuning_kws = {}
        if self.ax is None:
            self.plot_box_plot(**fine_tuning_kws)
        self.df["temp_group"] = self.df[self.group]
        mean_dif, df_res = self._calc_sig(group_order=self.order)  # Calc p-value

        # add p-values based on df_res

        for k, row in df_res.iterrows():
            # i j represents the order of groups
            i = self.order.index(str(row.values[0]))
            j = self.order.index(str(row.values[1]))

            tick_size, x_line, y_text = self.calc_tick_x_line(
                position_offset=self.max_value_on_scale * rate_position_offset_with_hue if hue else 0,
                i=i, j=j, k=k)
            self.draw_lines(i, j, tick_size, x_line)
            sig_text = self.p_value_sig(self.get_p_value(row), mean_dif.meandiffs[k - 1])
            self.draw_sig_level(sig_text, tick_size, x_line, y_text)

    def calc_sig_levels_hue(self, hue, hue_order: list[str] = None, show_group: bool = True,
                            fine_tuning_kws: dict = None):

        if hue not in self.df.columns:
            raise ValueError(f"{hue} not in columns")
        if not hue_order:
            hue_order = list(self.df[hue].unique())
        if fine_tuning_kws is None:
            fine_tuning_kws = {}
        self.plot_box_plot(hue=hue, hue_order=hue_order, **fine_tuning_kws)
        if show_group:
            self.calc_sig_levels_group(hue=True)
        # ['Female|Voice Female', 'Female|Chatbot Female', 'Female|filter', 'Male|Voice Female', 'Male|Chatbot Female', 'Male|filter']
        self.df["temp_group"] = self.df[self.group].astype(str) + "|" + self.df[hue].astype(str)
        group_order = [f"{i}|{j}" for i in self.order for j in hue_order]
        mean_dif, df_res = self._calc_sig(group_order=group_order, annotation=not show_group)  # Calc p-value
        from itertools import combinations

        # add p-values based on df_res
        for count_group, group in enumerate(self.order):
            group_start = 0.1 + count_group - 1 + 0.5
            k = 0
            for hue_1, hue_2 in [comb for comb in combinations(hue_order, 2)]:
                k += 1
                name_group_1 = f"{group}|{hue_1}"
                name_group_2 = f"{group}|{hue_2}"

                width_of_each_hue = 0.8 / len(hue_order)
                hue_1_mid = width_of_each_hue * hue_order.index(hue_1) + width_of_each_hue / 2
                hue_2_mid = width_of_each_hue * hue_order.index(hue_2) + width_of_each_hue / 2
                # based on hue order
                i = group_start + hue_1_mid
                j = group_start + hue_2_mid
                tick_size, x_line, y_text = self.calc_tick_x_line(fix_position=False, position_offset=0.3,
                                                                  i=i, j=j, k=k)
                self.draw_lines(i, j, tick_size, x_line)
                index = df_res[(df_res["group1"] == name_group_1) & (df_res["group2"] == name_group_2)].index
                index = list(index)[0] - 1
                sig_text = self.p_value_sig(float(df_res.iloc[index].pval), float(mean_dif.meandiffs[index]))
                self.draw_sig_level(sig_text, tick_size, x_line, y_text, fontsize=10)
