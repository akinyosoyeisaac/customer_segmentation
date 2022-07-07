import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import KernelPCA
import json
import pickle as pk
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")




def n_cluster_visual(path: str, y):

    plt.figure(figsize=(10,8))
    plt.plot(n_cluster_range, y)
    plt.grid(b=True, which="major", axis="both", color="green", linewidth=.5)
    plt.xlabel("N-CLUSTERS", size=15, weight="bold")
    if y[0] >= 1.1:
        plt.ylabel("INERTIA", size=15, weight="bold")
    else:
        plt.ylabel("silhouette".upper(), size=15, weight="bold")
    plt.title("No. Clusters Vs Inertia", size=22, weight="bold")
    plt.savefig(path);



with open("params.yaml") as file:
    config = yaml.safe_load(file)

df = pd.read_csv(config["data_loader"]["processed_data"])
X = df.copy().values

kpca = KernelPCA(n_components=2,kernel="rbf", random_state=config["random_state"])
X_kpca = kpca.fit_transform(X)

with open(config["data_loader"]["X_kpca"], "w") as file:
    pk.dump(X_kpca, file)

inertia = []
silhouette_scores = []
n_cluster_range = range(2, config["train"]["n_cluster_range"])

for x in n_cluster_range:
    clus_model = KMeans(n_clusters=x, random_state=config["random_state"])
    clus_model.fit(X_kpca)
    inertia.append(clus_model.inertia_)
    silhouette_scores.append(silhouette_score(df, clus_model.labels_, random_state=config["random_state"]))

hyper_metrics = {"inertia":inertia, "silhouette_scores": silhouette_scores}
with open(config["report"]["metrics"]["hyper_metrics"], "w") as file:
    json.dump(hyper_metrics, file)

n_cluster_visual(config["report"]["visual"]["no_clustersvsinertia"], y=inertia)

n_cluster_visual(config["report"]["visual"]["no_clustersvssilhouette"], y=silhouette_scores)