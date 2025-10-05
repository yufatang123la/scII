import os
import torch
import numpy as np
import scanpy as sc
import seaborn as sns
import scanpy.external as sce
from torch import nn
from model import feature_prototype_similarity, gmm

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def infer_result(net, scrna_dataloader, scatac_dataloader, args):
    net.eval()
    feature_vec, type_vec, pred_vec, loss_vec = [], [], [], []

    scatac_logits_vec = []
    for (x, y) in scrna_dataloader:
        x = x.cuda()
        with torch.no_grad():
            h = net.encoder(x)
        feature_vec.extend(h.cpu().numpy())
        type_vec.extend(y.numpy())
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    for (x, _), _ in scatac_dataloader:
        x = x.cuda()
        with torch.no_grad():
            h = net.encoder(x)
            logit= net.classifier(h)
            pred = torch.argmax(logit, dim=-1)
            loss = ce_loss(logit, pred)
        feature_vec.extend(h.cpu().numpy())
        pred_vec.extend(pred.cpu().numpy())
        loss_vec.extend(loss.cpu().numpy())
        scatac_logits_vec.extend(logit.cpu().numpy())
    feature_vec, type_vec, pred_vec, loss_vec, scatac_logits_vec= (
        np.array(feature_vec),
        np.array(type_vec),
        np.array(pred_vec),
        np.array(loss_vec),
        np.array(scatac_logits_vec)
    )

    similarity, _ = feature_prototype_similarity(
        feature_vec[: len(scrna_dataloader.dataset)],
        type_vec,
        feature_vec[len(scrna_dataloader.dataset) :],
    )


    similarity = np.array(similarity) if not isinstance(similarity, np.ndarray) else similarity
    loss_vec = np.array(loss_vec) if not isinstance(loss_vec, np.ndarray) else loss_vec

    #prob_feature = gmm(1 - similarity, metric_weights=[0.7, 0.3])
    #prob_logit = gmm(loss_vec)

    prob_feature = gmm(1 - similarity)

    prob_logit = gmm(loss_vec)
    prob_feature = np.array(prob_feature) if not isinstance(prob_feature, np.ndarray) else prob_feature
    prob_logit = np.array(prob_logit) if not isinstance(prob_logit, np.ndarray) else prob_logit

    reliability_vec = prob_feature * prob_logit

    if args.novel_type:
        prob_gmm = gmm(reliability_vec)
        novel_index = prob_gmm > 0.5
        pred_vec[novel_index] = -1

    return feature_vec, pred_vec, reliability_vec,scatac_logits_vec

def plot_cell_type_probabilities(scatac_logits, scatac_true_labels, label_map, args, accuracy=None):
    probabilities = torch.softmax(torch.from_numpy(scatac_logits), dim=1).numpy()

    num_cell_types = probabilities.shape[1]
    cell_type_names = [label_map[i] for i in range(num_cell_types)]
    prediction_probability_matrix = np.zeros((num_cell_types, num_cell_types))
    counts_per_true_type = np.zeros(num_cell_types)
    if scatac_true_labels is not None:
        for i in range(scatac_true_labels.shape[0]):
            true_label = scatac_true_labels[i]
            if true_label != -1:
                prediction_probability_matrix[true_label, :] += probabilities[i, :]
                counts_per_true_type[true_label] += 1
        for i in range(num_cell_types):
            if counts_per_true_type[i] > 0:
                prediction_probability_matrix[i, :] /= counts_per_true_type[i]

    plt.figure(figsize=(4, 4))

    ax = sns.heatmap(
        prediction_probability_matrix,
        annot=False,
        cmap="Reds",
        fmt=".2f",
        xticklabels=cell_type_names,
        yticklabels=cell_type_names,
        vmin=0.0,
        vmax=1.0,
        #cbar=False,
        linewidths=0.5,
        linecolor='lightgray',
        square=True,
        cbar=True,
        cbar_kws={
            "label": "Fractions of agreement",
            "shrink": 0.7,
            "ticks": [0, 0.25, 0.50, 0.75, 1.00] ,
            "extend": 'neither',
            "fraction": 0.046,
            "pad": 0.04
        }
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label("Fractions of agreement", fontname='Times New Roman')

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_color('black')


    cbar.ax.yaxis.set_tick_params(direction='in', color='black', length=3)

    font_name_to_set = 'Times New Roman'

    ax.tick_params(axis='both', which='both', length=0)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.setp(ax.get_xticklabels(), fontname=font_name_to_set)
    plt.setp(ax.get_yticklabels(), fontname=font_name_to_set)

    plt.xlabel("Predicted cell type", fontname=font_name_to_set,fontsize=12)
    plt.ylabel("True cell type", fontname=font_name_to_set,fontsize=12)
    title_text = "Kidney: "
    if accuracy is not None:
        title_text += f"(ACC: {accuracy * 100:.0f}%)"
    plt.title(title_text, fontname=font_name_to_set,fontsize=14)

    os.makedirs("Frameworkfig/", exist_ok=True)
    plt.savefig(
        f"Frameworkfig/" + "scII" + "_CellTypeProbability.tif",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=300
    )
    plt.close()

def save_result(
    feature_vec,
    pred_vec,
    reliability_vec,
    label_map,
    type_num,
    scrna_adata,
    scatac_adata,
    args,
    scatac_logits_vec=None
):
    adata = sc.AnnData(feature_vec)
    adata.obs["Domain"] = np.concatenate(
        (scrna_adata.obs["Domain"], scatac_adata.obs["Domain"]), axis=0
    )
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(adata, "Domain", theta=0.0, verbose=False)
    feature_vec = adata.obsm["X_pca_harmony"]

    scrna_adata.obsm["Embedding"] = feature_vec[: len(scrna_adata.obs["Domain"])]
    scatac_adata.obsm["Embedding"] = feature_vec[len(scrna_adata.obs["Domain"]) :]
    predictions = np.empty(len(scatac_adata.obs["Domain"]), dtype=np.dtype("U30"))
    for k in range(type_num):
        predictions[pred_vec == k] = label_map[k]
    if args.novel_type:
        predictions[pred_vec == -1] = "Novel (Most Unreliable)"
    scatac_adata.obs["Prediction"] = predictions
    scatac_adata.obs["Reliability"] = reliability_vec
    # --- begin---
    try:
        scatac_label_int = torch.from_numpy(
            (
                    scatac_adata.obs["cell_type"]
                    .rank(method="dense", ascending=True)
                    .astype(int)
                    - 1
            ).values
        ).numpy()
        evaluation = True
    except:
        print("未提供目标细胞类型注释，跳过评估和概率图绘制")
        evaluation = False
        scatac_label_int = None

    if evaluation and scatac_logits_vec is not None:
        acc_for_plot = accuracy_score(scatac_label_int, pred_vec)
        plot_cell_type_probabilities(
            scatac_logits_vec, scatac_label_int, label_map, args, accuracy=acc_for_plot
        )
    # --- over ---

    if args.umap_plot:
        sc.set_figure_params(figsize=(4, 4), dpi=300)
        try:
            scatac_annotation = scatac_adata.obs["cell_type"]
        except:
            scatac_annotation = scatac_adata.obs["Prediction"]
        adata.obs["cell_type"] = np.concatenate(
            (scrna_adata.obs["cell_type"], scatac_annotation), axis=0
        )
        os.makedirs("Frameworkfig/", exist_ok=True)
        print("visualization...")
        sc.pp.neighbors(adata, use_rep="X_pca_harmony")
        sc.tl.umap(adata)
        umap_coords = adata.obsm['X_umap']
        mean_coords = umap_coords.mean(axis=0)

        adata.obsm['X_umap'] = umap_coords - mean_coords
        #hls_colors = sns.color_palette("hls", n_colors=len(all_cell_types_sorted))
        #cell_type_palette_dict = {
           # cell_type: color for cell_type, color in zip(all_cell_types_sorted, hls_colors)
        #}
        cell_types = sorted(adata.obs["cell_type"].unique())

        palette = sns.color_palette("tab20c", len(cell_types))
#------cluster-------
        #  UMAP
        sc.pl.umap(
            adata,
            color=["cell_type"],
            palette=palette,
            size=2,
            title="Kidney",
            frameon=True,
            legend_loc=None,
            alpha=0.8,
            show=False,
        )

        ax = plt.gca()
        ax.grid(False)
        ax.set_xlabel("UMAP1", fontsize=10)
        ax.set_ylabel("UMAP2", fontsize=10)

        # 1. Obtain the range of the coordinate axes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # 2. max
        max_abs_val = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

        # 4. Symmetry about 0
        ax.set_xlim(-max_abs_val, max_abs_val)
        ax.set_ylim(-max_abs_val, max_abs_val)

        # 5. 1:1
        ax.set_aspect('equal', adjustable='box')

        # The displayed scale
        ticks_to_show = [-10, 0, 10]

        ax.set_xticks(ticks_to_show)
        ax.set_yticks(ticks_to_show)

        ax.tick_params(axis='both', which='major', labelsize=10)

        # Create custom legends (with borders)
        legend_elements = [
            plt.Line2D([0], [0],
                       marker="o",
                       color="w",
                       label=cell_type,
                       markerfacecolor=palette[i],
                       markersize=5)
            for i, cell_type in enumerate(cell_types)
        ]

        #legend upper left
        plt.legend = ax.legend(
            handles=legend_elements,
            title="Cell Type",
            loc="upper left",
            bbox_to_anchor=(1.0, 1),
            frameon=True,
            edgecolor="lightgray",
            facecolor="white",
            fontsize=8,
        )
        plt.setp(plt.gca().get_legend().get_title(), fontsize=10)

        plt.savefig(
            "Frameworkfig/" + "scII" + "_Cluster.tif",
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300
        )
        plt.close()

        batch_palette =["#FFD0A5", "#95B1CE"]
        sc.pl.umap(
            adata,
            color=["Domain"],
            palette=batch_palette,
            size=2,
            title='Kidney',
            frameon=True,
            legend_loc=None,
            alpha=0.8,
            show=False,
        )

        ax = plt.gca()
        ax.grid(False)
        ax.set_xlabel("UMAP1", fontsize=10)
        ax.set_ylabel("UMAP2", fontsize=10)


        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()


        max_abs_val = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))


        ax.set_xlim(-max_abs_val, max_abs_val)
        ax.set_ylim(-max_abs_val, max_abs_val)

        ax.set_aspect('equal', adjustable='box')
        ticks_to_show = [-10, 0, 10]
        ax.set_xticks(ticks_to_show)
        ax.set_yticks(ticks_to_show)

        ax.tick_params(axis='both', which='major', labelsize=10)

        # 添加自定义Domain图例
        domain_labels = adata.obs["Domain"].unique()
        domain_labels = [label for label in ["scATAC", "scRNA"] if label in domain_labels]
        # domain_palette = sns.color_palette("hls", 2)
        legend_elements = [
            plt.Line2D([0], [0],
                       marker="o",
                       color="w",
                       label=domain,
                       markerfacecolor=batch_palette[i],
                       markersize=5)
            for i, domain in enumerate(domain_labels)
        ]

        legend = ax.legend(
            handles=legend_elements,
            title="Batch",
            loc="upper right",
            bbox_to_anchor=(1.22, 1),
            frameon=True,
            edgecolor="lightgray",
            facecolor="white",
            fontsize=8,
        )
        plt.setp(legend.get_title(), fontsize=10)
        #"""
        plt.savefig(
            "Frameworkfig/" + "scII" + "_batch.tif",
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300
        )
        plt.close()
    try:
        scatac_label_int = torch.from_numpy(
            (
                scatac_adata.obs["cell_type"]
                .rank(method="dense", ascending=True)
                .astype(int)
                - 1
            ).values
        )
        evaluation = True
    except:
        print("No Cell Type ")
        evaluation = False

    if evaluation and not args.novel_type:
        print("=======Eval=======")

        count = torch.unique(scatac_label_int, return_counts=True, sorted=True)[1].float()
        f1_weight = 1.0 / (count + 1e-6)
        f1_weight = f1_weight / f1_weight.sum()
        f1_weight = f1_weight.numpy()


        scaler = MinMaxScaler(feature_range=(0.5, 1))
        f1_weight = scaler.fit_transform(f1_weight.reshape(-1, 1)).flatten()
        f1_weight = f1_weight / f1_weight.sum()
        acc = accuracy_score(scatac_label_int, pred_vec)
        f1_scores = f1_score(scatac_label_int, pred_vec, average=None)
        f1_scores = np.clip(f1_scores, 0, 1)
        #  F1-score
        f1_acc = (f1_scores * f1_weight).sum()
        f1_acc = np.clip(f1_acc, 0, 1)

        # --- NMI, ARI, AMI  ---
        nmi = normalized_mutual_info_score(scatac_label_int, pred_vec)
        ari = adjusted_rand_score(scatac_label_int, pred_vec)
        ami = adjusted_mutual_info_score(scatac_label_int, pred_vec)
        # ---  NMI, ARI, AMI---
        print("ACC: %.4f, F1-score: %.4f" % (acc, f1_acc))
        print("NMI: %.4f, ARI: %.4f, AMI: %.4f" % (nmi, ari, ami))
        if args.umap_plot:
            Stype = silhouette_score(
                adata.obsm["X_umap"], adata.obs["cell_type"].values
            )
            Somic = silhouette_score(
                adata.obsm["X_umap"], adata.obs["Domain"].values
            )
            HSC = (2* (1 - (Somic + 1) / 2)* (Stype + 1)/ 2/ (1 - (Somic + 1) / 2 + (Stype + 1) / 2))
            print("HSC: %.4f"% (HSC))
    print(
        "Successful integration"
    )
