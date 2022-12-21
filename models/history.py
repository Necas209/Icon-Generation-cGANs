from dataclasses import dataclass, field
from typing import List


@dataclass
class History:
    """ History of training """
    d_loss: List[float] = field(default_factory=list)
    d_acc: List[float] = field(default_factory=list)
    g_loss: List[float] = field(default_factory=list)

    def add(self, d_loss: float, d_acc: float, g_loss: float) -> None:
        self.d_loss.append(d_loss)
        self.d_acc.append(d_acc)
        self.g_loss.append(g_loss)


def plot_history(history: History) -> None:
    """ Plot the history of training """
    import matplotlib.pyplot as plt
    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.d_loss, label='discriminator')
    plt.plot(history.g_loss, label='generator')
    plt.legend()
    plt.title('Loss')
    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history.d_acc, label='discriminator')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
