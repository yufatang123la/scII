import torch
import random
import argparse
import time
import numpy as np
from copy import deepcopy
from model import Net
from plot import infer_result, save_result
from process import prepare_dataloader, partition_data, adjacency
print(torch.cuda.is_available())
def main(args):
    start_time = time.time()
    (
        scrna_dataset,
        scrna_dataloader_train,
        scrna_dataloader_eval,
        scatac_dataset,
        scatac_dataloader_train,
        scatac_dataloader_eval,
        gene_num,
        type_num,
        label_map,
        scrna_adata,
        scatac_adata,
    ) = prepare_dataloader(args)

    scrna_dataloader_eval_all = deepcopy(scrna_dataloader_eval)
    scatac_dataloader_eval_all = deepcopy(scatac_dataloader_eval)
    if args.novel_type:
        scatac_adj = adjacency(scatac_dataset.tensors[0])
    else:
        scatac_adj = None

    scrna_label = scrna_dataset.tensors[1]
    count = torch.unique(scrna_label, return_counts=True, sorted=True)[1]
    ce_weight = 1.0 / count
    ce_weight = ce_weight / ce_weight.sum() * type_num
    ce_weight = ce_weight.cuda()

    print("======= demo_begin =======")

    net = Net(gene_num, type_num, ce_weight, args).cuda()
    preds, prob_feat, prob_logit = net.run(
        scrna_dataloader_train,
        scrna_dataloader_eval,
        scatac_dataloader_train,
        scatac_dataloader_eval,
        scatac_adj,
        args,
    )


    for iter in range(args.max_iteration):
        (
            scrna_dataloader_train,
            scrna_dataloader_eval,
            scatac_dataloader_train,
            scatac_dataloader_eval,
            scrna_dataset,
            scatac_dataset,
        ) = partition_data(
            preds,
            prob_feat,
            prob_logit,
            scrna_dataset,
            scatac_dataset,
            args,
        )


        if scatac_dataset.__len__() <= args.batch_size:
            break
        print("======= iter:", iter + 1, "=======")

        scrna_label = scrna_dataset.tensors[1]
        count = torch.unique(scrna_label, return_counts=True, sorted=True)[1]
        ce_weight = 1.0 / count
        ce_weight = ce_weight / ce_weight.sum() * type_num
        ce_weight = ce_weight.cuda()

        net = Net(gene_num, type_num, ce_weight, args).cuda()
        preds, prob_feat, prob_logit = net.run(
            scrna_dataloader_train,
            scrna_dataloader_eval,
            scatac_dataloader_train,
            scatac_dataloader_eval,
            scatac_adj,
            args,
        )
    print("======= demo_over =======")

    features, predictions, reliabilities , scatac_logits = infer_result(
        net, scrna_dataloader_eval_all, scatac_dataloader_eval_all, args
    )
    save_result(
        features,
        predictions,
        reliabilities,
        label_map,
        type_num,
        scrna_adata,
        scatac_adata,
        args,
        scatac_logits_vec=scatac_logits,  #
    )
    # Total run time
    end_time = time.time()
    total_run_time = end_time - start_time
    print(f"Total run time: {total_run_time:.2f} seconds")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    start_time = time.time()
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="dual",
        choices=["dual", "similarity", "mlp"],
        help="dual: use both S^a and MLP loss; similarity: S^a only; mlp: MLP loss only."
    )
    # Data configs
    parser.add_argument("--data_path", type=str, default="test/") #celltype
    parser.add_argument("--scrna_data", type=str, default="scRNA.h5ad") #[9134 , 16750]
    parser.add_argument("--scatac_data", type=str, default="scATAC.h5ad") #[9134 , 16750]

    parser.add_argument("--scrna_preprocess", type=str)
    parser.add_argument("--scatac_preprocess", type=str)
    #parser.add_argument('--triplet_margin', type=float, default=1.0, help='Margin for triplet loss')

    # Model configs
    parser.add_argument("--reliability_threshold", default=0.95, type=float)
    parser.add_argument("--align_loss_epoch", default=1, type=float)
    parser.add_argument("--prototype_momentum", default=0.9, type=float)
    parser.add_argument("--early_stop_acc", default=0.99, type=float)
    parser.add_argument("--max_iteration", default=1, type=int)
    parser.add_argument("--novel_type", action="store_true")
    # Training configs
   
    #parser.add_argument('--domain_loss_weight', type=float, default=0.3, help="Weight for domain alignment loss.")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--train_epoch", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--random_seed", default=2025, type=int)
    # Evaluation configs
    parser.add_argument("--umap_plot", action="store_true",default="--umap_plot")

    args = parser.parse_args()
    #print("data_path:", args.data_path)
    #print("scrna_data:", args.scrna_data)
    #print("scatac_data:", args.scatac_data)

    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


    main(args)
