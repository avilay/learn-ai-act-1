import os
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts


def main():
    print("Running mlflow_tracking.py")

    log_param("param1", randint(0, 1000))

    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test.txt", "wt") as f:
        print("hello world!", file=f)

    log_artifacts("outputs")


main()
