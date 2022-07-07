import argparse
import pandas as pd
import yaml


def raw_data_loader(path: str):

    with open(path) as file:
        config = yaml.safe_load(file)

    df = pd.read_csv(config["data_loader"]["raw_data"], index_col="CUST_ID")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "-p", dest = "path", type=str, required=True)
    args = parser.parse_args()

    