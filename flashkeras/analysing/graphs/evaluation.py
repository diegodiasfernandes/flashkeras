from flashkeras.utils.otherimports import *

def plot_history_train_curve(history):
    """Plot training and validation loss and accuracy from a Keras history.

    `Flash Explanation:` *`Use this to assess learning progress and detect overfitting visually.`*
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


