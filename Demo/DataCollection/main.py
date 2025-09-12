import os
import time

import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt

def get_newest_filepath(base_path: pathlib.Path) -> pathlib.Path:
    items = os.listdir(base_path)
    sorted_items = sorted(items)
    return base_path / sorted_items[-1]

def get_pd(file_name: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(file_name / "measurement.csv")
    return df

def plot(df: pd.DataFrame):
    df = df.reset_index()
    ax = plt.figure(figsize=(10, 6))
    sns.lineplot(df, x="index", y="x")
    sns.lineplot(df, x="index", y="y")
    sns.lineplot(df, x="index", y="z")
    plt.suptitle("Acceraleration in G-Values")
    plt.savefig("plot.png")
    plt.show()


if __name__ == '__main__':
    last_folder = ""
    while True:

        base_path = pathlib.Path("/Users/leo/work/Predictor/cloud/elastic-ai.runtime.cloud-main/applications/ballChallenge/SensorValues")
        new_throw_folder = get_newest_filepath(base_path)

        if os.path.exists(new_throw_folder / "measurement.csv"):
            if last_folder != str(new_throw_folder):
                last_folder = str(new_throw_folder)
                plt.close('all')
                my_pd = get_pd(new_throw_folder.absolute())
                plot(my_pd)
        time.sleep(0.1)