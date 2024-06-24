import warnings

def formatting(text):
    return text.replace("_", " ").title()


def renaming(ax):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Major Label
        ax.set_ylabel(formatting(ax.get_ylabel()))
        ax.set_xlabel(formatting(ax.get_xlabel()))

        # Tick Label
        labels = [item.get_text() for item in ax.get_yticklabels()]
        for i, label in enumerate(labels):
            labels[i] = formatting(label)
        ax.set_yticklabels(labels)

        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i, label in enumerate(labels):
            labels[i] = formatting(label)
        ax.set_xticklabels(labels)

        # Legend
        handles, labels = [t for t in ax.get_legend_handles_labels()]
        if labels:
            for i, label in enumerate(labels):
                labels[i] = formatting(label)
            ax.legend(handles=handles, labels=labels, title=formatting(ax.axes.get_legend().get_title().get_text()))

    return ax

