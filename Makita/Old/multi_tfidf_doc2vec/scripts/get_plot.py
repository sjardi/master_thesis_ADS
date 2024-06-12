"""Script to get dataset sorted on the average time to discovery.

Example
-------

> python scripts/get_plot.py

or

> python scripts/get_plot.py -s asreview_files -o plot.png


Authors
-------
- Teijema, Jelle
"""

# version 0.0.0

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from asreview import open_state

from asreviewcontrib.insights.plot import plot_recall


def _set_legend(ax, state, legend_option, label_to_line, state_file):
    metadata = state.settings_metadata
    label = None

    if legend_option == "filename":
        label = state_file.stem
    elif legend_option == "model":
        label = " - ".join(
            [metadata["settings"]["model"],
             metadata["settings"]["feature_extraction"],
             metadata["settings"]["balance_strategy"],
             metadata["settings"]["query_strategy"]])
    elif legend_option == "classifier":
        label = metadata["settings"]["model"]
    else:
        try:
            label = metadata["settings"][legend_option]
        except KeyError as err:
            raise ValueError(f"Invalid legend setting: '{legend_option}'") from err  # noqa: E501

    if label:
        # add label to line
        if label not in label_to_line:
            ax.lines[-2].set_label(label)
            label_to_line[label] = ax.lines[-2]
        # set color of line to the color of the first line with the same label
        else:
            ax.lines[-2].set_color(label_to_line[label].get_color())
            ax.lines[-2].set_label("_no_legend_")


def get_plot_from_states(states, filename, legend=None):
    """Generate an ASReview plot from state files.

    Arguments
    ---------
    states: list
        List of state files.
    filename: str
        Filename of the plot.
    legend: str
        Add a legend to the plot, based on the given parameter.
        Possible values: "filename", "model", "feature_extraction",
        "balance_strategy", "query_strategy", "classifier".
    """
    states = sorted(states)
    fig, ax = plt.subplots()
    label_to_line = {}

    for state_file in states:
        with open_state(state_file) as state:
            plot_recall(ax, state)
            if legend:
                _set_legend(ax, state, legend, label_to_line, state_file)

    if legend:
        ax.legend(loc=4, prop={'size': 8})
    fig.savefig(str(filename))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate an ASReview plot from the found state files."
    )
    parser.add_argument(
        "-s",
        type=str,
        help="States location")
    parser.add_argument(
        "-o",
        type=str,
        help="Output location")
    parser.add_argument(
        "--show_legend", "-l",
        type=str,
        help="Add a legend to the plot, based on the given parameter.")
    args = parser.parse_args()

    # load states
    states = list(Path(args.s).glob("*.asreview"))

    # check if states are found
    if len(states) == 0:
        raise FileNotFoundError(f"No state files found in {args.s}")

    # generate plot and save results
    get_plot_from_states(states, args.o, args.show_legend)