"""DQN.

Usage:
  dqn train ENV --hparams=<hparams>
  dqn play ENV [--qnet=<qnet>]
  dqn (-h | --help)

Arguments:
  ENV   Environment name.

Options:
  -h --help                 Show this screen.
  --hparams=<hparams>       Hyper parameters file in .ini format.
  --qnet=<qnet>             Weights file of the trained Q-network.
"""

from docopt import docopt


def main():
    args = docopt(__doc__, version="DQN 1.0")
    print(args)


if __name__ == "__main__":
    main()
