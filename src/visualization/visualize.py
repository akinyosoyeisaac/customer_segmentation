# importing the neccessary labraries
import pandas as pd
import yaml
import argparse
from sklearn.cluster import KMeans
import json
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-dark")


def feature_impt(cluster_centers, config):
    """
    input: 
        1. cluster_centers[arrays]
        2. configuration file

    output: jpeg of the barplot showing the feature importance in an ascending order
    """
    plt.figure(figsize=(10,8))
    cluster_centers[:config["train"]["n_impt_feat"]].plot.bar()
    plt.xticks(rotation=90)
    plt.xlabel("LABELS", size=15, weight="bold")
    plt.ylabel("FEATURE IMPORTANCES", size=15, weight="bold")
    plt.title("FEATURE IMPORTANCES USING CLUSTER CENTERS", size=22, weight="bold")
    plt.tight_layout()
    plt.savefig(config["report"]["visual"]["feature_importance"]);

def cluster_center(model, config, X_kpca):
    """
    input: 
        1. model trained on the two dimensional array
        2. configuration file
        3. our 2D data
        
    output: jpeg of the scatterplot and the cluster centers
    """
    color=["red", "black", "orange", "yellow", "green"]
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X_kpca[:, 0], y=X_kpca[:, 1])
    plt.scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], s=250, c=color)
    plt.title("CLUSTER POINT ON 2D SCALE", size=22, weight="bold")
    plt.xlabel("KPCA1", size=15, weight="bold")
    plt.xlabel("KPCA1", size=15, weight="bold")
    plt.savefig(config["report"]["visual"]["cluster_center"]);

def visualization(config_file:str):
    # loading of configuration files
    with open(config_file) as file:
        config = yaml.safe_load(file)


    # loading the featurized data
    df = pd.read_csv(config["data_loader"]["processed_data"])

    # loading the mean cluster center generated at the training stage
    with open(config["report"]["metrics"]["feature_importance"], "r") as file:
        mean_cluster_center = json.load(file)

    # Visualizing the first ten of most importance features
    feature_impt(mean_cluster_center, config)

    # loading the data with dimension decomposition
    with open(config["data_loader"]["X_kpca"], "rb") as file:
        X_kpca = pk.load(file)

    # runing the kmeans algorithm on the data above
    model_vis = KMeans(n_clusters=config["train"]["n_cluster"], random_state=config["random_state"])
    model_vis.fit(X_kpca)

    # visualizing the cluster centers on a 2D array data
    cluster_center(model_vis, config, X_kpca)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "--p", dest = "path", type=str, required=True)
    args = parser.parse_args()
    visualization(args.path)


