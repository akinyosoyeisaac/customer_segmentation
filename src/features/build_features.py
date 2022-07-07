import argparse
import pandas as pd
import yaml
from src.data.raw_data_loader import raw_data_loader
from sklearn.preprocessing import PowerTransformer

    
    
def featurization(path: str):

    with open(path) as file:
        config = yaml.safe_load(file)

    df = raw_data_loader(path)
    df.fillna(0, inplace=True)
    transformer = PowerTransformer()
    df = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)

    df.to_csv(config["data_loader"]["processed_data"], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "--p", dest = "path", type=str, required=True)
    args = parser.parse_args()
    featurization(args.path)