import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

import logging

logformat = "[%(levelname)s %(asctime)s] %(process)s-%(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logformat, datefmt="%m-%d %I:%M:%S")

logging.getLogger("git.cmd").setLevel(logging.ERROR)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main():
    np.random.seed(40)
    datafile = os.path.expanduser("~/mldata/winequality-red.csv")
    data = pd.read_csv(datafile, sep=";")

    # Split the data into 75% 25% split
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"\tRMSE: {rmse:.3f}")
        print(f"\tMAE: {mae:.3f}")
        print(f"\tR2: {r2:.3f}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", np.around(rmse, 3))
        mlflow.log_metric("r2", np.around(r2, 3))
        mlflow.log_metric("mae", np.around(mae, 3))

        mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    main()
