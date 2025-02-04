from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class History:
    """ History of training """
    d_loss: list[float] = field(default_factory=list)
    d_acc: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)

    def add(self, d_loss: float, d_acc: float, g_loss: float) -> None:
        """ Add a new entry to the history """
        self.d_loss.append(d_loss)
        self.d_acc.append(d_acc)
        self.g_loss.append(g_loss)

    def plot(self) -> None:
        """ Plot the history """
        plt.plot(self.d_loss, label='discriminator')
        plt.plot(self.g_loss, label='generator')
        plt.legend()
        plt.title('Loss')
        plt.show()

        plt.plot(self.d_acc, label='discriminator')
        plt.legend()
        plt.title('Accuracy')
        plt.show()
