from flashkeras.utils.otherimports import *

def plot_history_train_curve(history):
    """Plot training and validation curves for every metric in a Keras history.

    `Flash Explanation:` *`Use this to assess learning progress and detect overfitting visually.`*
    """
    history_data = history.history if hasattr(history, "history") else history
    metric_names = [name for name in history_data if not name.startswith("val_")]
    metric_names.sort(key=lambda name: (name != "loss", name))

    if not metric_names:
        raise ValueError("history must contain at least one training metric")

    columns = min(2, len(metric_names))
    rows = (len(metric_names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False)
    axes = axes.ravel()

    for axis, metric_name in zip(axes, metric_names):
        axis.plot(history_data[metric_name], label="train")
        validation_name = f"val_{metric_name}"
        if validation_name in history_data:
            axis.plot(history_data[validation_name], label="val")
        axis.set_title(metric_name.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.legend()

    for axis in axes[len(metric_names):]:
        axis.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


