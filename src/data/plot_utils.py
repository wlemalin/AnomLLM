import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def create_color_generator(exclude_color='blue'):
    # Get the default color list
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
    # Filter out the excluded color
    filtered_colors = [
        color for color in default_colors if color != exclude_color]
    # Create a generator that yields colors in order
    return (color for color in filtered_colors)


def plot_series_and_predictions(
    series: np.ndarray,
    gt_anomaly_intervals: list[list[tuple[int, int]]],
    anomalies: Optional[dict] = None,
    single_series_figsize: tuple[int, int] = (20, 3),
    gt_ylim: tuple[int, int] = (-1, 1),
    gt_color: str = 'steelblue',
    anomalies_alpha: float = 0.5
) -> None:
    plt.figure(figsize=single_series_figsize)

    color_generator = create_color_generator()

    def get_next_color(color_generator):
        try:
            # Return the next color
            return next(color_generator)
        except StopIteration:
            # If all colors are used, reinitialize the generator and start over
            color_generator = create_color_generator()
            return next(color_generator)

    num_anomaly_methods = len(anomalies) if anomalies else 0
    ymin_max = [
        (
            i / num_anomaly_methods * 0.5 + 0.25,
            (i + 1) / num_anomaly_methods * 0.5 + 0.25,
        )
        for i in range(num_anomaly_methods)
    ]
    ymin_max = ymin_max[::-1]

    for i in range(series.shape[1]):
        plt.ylim(gt_ylim)
        plt.plot(series[:, i], color=gt_color)

        if gt_anomaly_intervals is not None:
            for start, end in gt_anomaly_intervals[i]:
                plt.axvspan(start, end, alpha=0.2, color=gt_color)

        if anomalies is not None:
            for idx, (method, anomaly_values) in enumerate(anomalies.items()):
                if anomaly_values.shape == series.shape:
                    anomaly_values = np.nonzero(
                        anomaly_values[:, i])[0].flatten()
                ymin, ymax = ymin_max[idx]
                # Use the function to get a random color
                random_color = get_next_color(color_generator)
                for anomaly in anomaly_values:
                    plt.axvspan(anomaly, anomaly + 1, ymin=ymin, ymax=ymax,
                                alpha=anomalies_alpha, color=random_color, lw=0)
                plt.plot([], [], color=random_color, label=method)

    plt.tight_layout()
    if anomalies is not None:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return plt.gcf()
