import pandas as pd
import argparse
import yaml
import json
from sklearn.cluster import KMeans
import pickle as pk


def training(config_path:str):

    # loading of configuration files
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # loading the featurized data 
    df = pd.read_csv(config["data_loader"]["processed_data"])

    # converting the data above into a numpy array
    X = df.copy().values


    # initializing our kmeans model
    model = KMeans(n_clusters=config["train"]["n_cluster"], random_state=config["random_state"])
    model.fit(X) # fitting our model on the data

    # extracting our mean of the cluster center
    mean_cluster_center = pd.Series(model.cluster_centers_.mean(0), index=df.columns).sort_values(ascending=False)

    # dumping of mean_cluster_centers into file for versoning
    with open(config["report"]["metrics"]["feature_importance"], "w") as file:
        json.dump(mean_cluster_center.to_dict(), file)

    # dumping of model into file for versoning
    with open(config["train"]["model"], "wb") as  file:
        pk.dump(model, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "--p", dest = "path", type=str, required=True)
    args = parser.parse_args()
    training(args.path)

