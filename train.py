import os
import csv
import copy
import random
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from logger import CompleteLogger
from dataload_rarp import CustomVideoDataset
from baseline.SEDMamba import MultiStageModel

# Initialize worker seed for reproducibility
def worker_init_fn(num_workers, rank, seed):
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# Train the model
def train_model(args, data_split_train_path, data_split_test_path):
    criterion = nn.BCEWithLogitsLoss().to(device)
    model = MultiStageModel(args.num_block, args.com_factor, args.features_dim, args.num_class)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_dataset = CustomVideoDataset(data_split_train_path)
    test_dataset = CustomVideoDataset(data_split_test_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.work,
        worker_init_fn=worker_init_fn(args.work, 0, args.seed),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.work,
        worker_init_fn=worker_init_fn(args.work, 0, args.seed),
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_AUC = 0.0
    best_test_mAP = 0.0
    best_epoch = 0

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        train_all_scores = []
        train_all_preds = []
        train_all_labels = []

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            video_fe, vl, e_labels = data[0].to(device), data[1], data[2].squeeze(0).to(device)
            video_fe = video_fe.transpose(2, 1)

            predictions = model.forward(video_fe)
            predictions = predictions.squeeze()
            loss = criterion(predictions, e_labels.float())
            scores = torch.sigmoid(predictions)
            preds = torch.round(scores)

            loss.backward()
            optimizer.step()

            train_all_scores.extend(scores.flatten().tolist())
            train_all_preds.extend(preds.flatten().tolist())
            train_all_labels.extend(e_labels.flatten().tolist())

            train_loss += loss.data.item()

        train_average_loss = float(train_loss) / len(train_dataloader)
        train_AUC = roc_auc_score(train_all_labels, train_all_scores)
        train_mAP = average_precision_score(train_all_labels, train_all_scores)

        model.eval()
        test_all_scores = []
        test_all_preds = []
        test_all_labels = []
        test_each_vidoe_names = []
        test_video_lengths = []

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                video_fe, vl, e_labels, video_name = data[0].to(device), data[1], data[2].squeeze(0), data[3]
                video_fe = video_fe.transpose(2, 1)

                predictions = model.forward(video_fe)
                predictions = predictions.squeeze()
                test_scores = torch.sigmoid(predictions)
                test_preds = torch.round(test_scores)

                test_all_scores.extend(test_scores.flatten().tolist())
                test_all_preds.extend(test_preds.flatten().tolist())
                test_all_labels.extend(e_labels.flatten().tolist())

                test_each_vidoe_names.append(video_name[0])
                test_video_lengths.append(int(vl.data[0]))

        test_AUC = roc_auc_score(test_all_labels, test_all_scores)
        test_mAP = average_precision_score(test_all_labels, test_all_scores)

        print(
            "epoch: {}"
            " train loss: {:4.4f}"
            " train AUC: {:4f}%"
            " train mAP: {:4f}%"
            " test AUC: {:4f}%"
            " test mAP: {:4f}%".format(
                epoch,
                train_average_loss,
                train_AUC * 100,
                train_mAP * 100,
                test_AUC * 100,
                test_mAP * 100,
            )
        )

        if test_AUC > best_test_AUC or (test_AUC == best_test_AUC and test_mAP > best_test_mAP):
            best_test_AUC = test_AUC
            best_test_mAP = test_mAP
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            base_name = str(args.exp) + "_best"
            if not os.path.exists("./exp_log/{}/{}/".format(args.lr, args.exp)):
                os.makedirs("./exp_log/{}/{}/".format(args.lr, args.exp))
            torch.save(
                best_model_wts,
                "./exp_log/{}/{}/".format(args.lr, args.exp) + base_name + ".pth",
            )
            print("updated best model: {}, AUC: {}".format(best_epoch, best_test_AUC))

    print("best_epoch", str(best_epoch))

    return best_test_mAP, best_test_AUC, test_all_preds, test_all_scores, test_all_labels, test_each_vidoe_names, test_video_lengths

# Main function
def main(args):
    root_data_path = args.data_path
    data_split_train_path = root_data_path + "/train_emb_DINOv2/"
    data_split_test_path = root_data_path + "/test_emb_DINOv2/"

    best_test_mAP, best_test_AUC, test_all_preds, test_all_scores, test_all_labels, test_each_vidoe_names, test_video_lengths = train_model(args, data_split_train_path, data_split_test_path)

    # Save the predictions, scores, and labels for each video
    start_idx = 0
    for i in range(len(test_each_vidoe_names)):
        preds_filename = "./exp_log/{}/{}/".format(args.lr, args.exp) + test_each_vidoe_names[i].split(".")[0] + ".csv"
        score_filename = "./exp_log/{}/{}/".format(args.lr, args.exp) + test_each_vidoe_names[i].split(".")[0] + "_score.csv"
        label_filename = "./exp_log/{}/{}/".format(args.lr, args.exp) + test_each_vidoe_names[i].split(".")[0] + "_label.csv"
        with open(preds_filename, "w") as f:
            writer = csv.writer(f)
            for j in range(test_video_lengths[i]):
                writer.writerow([test_all_preds[start_idx + j]])
        with open(score_filename, "w") as f:
            writer = csv.writer(f)
            for j in range(test_video_lengths[i]):
                writer.writerow([test_all_scores[start_idx + j]])
        with open(label_filename, "w") as f:
            writer = csv.writer(f)
            for j in range(test_video_lengths[i]):
                writer.writerow([test_all_labels[start_idx + j]])
        start_idx += test_video_lengths[i]

    print("best_test_mAp: {:f}".format(best_test_mAP * 100))
    print("best_test_AUC: {:f}".format(best_test_AUC * 100))

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED")
    parser.add_argument("-exp", default="SEDMamba", type=str, help="exp name")
    parser.add_argument("-dp", "--data_path", default="/path/to/your/data", type=str, help="path to data")
    parser.add_argument("-gpu_id", type=str, nargs="?", default="cuda:0", help="device id to run")
    parser.add_argument("-w", "--work", default=4, type=int, help="num of workers to use")
    parser.add_argument("-s", "--seed", default=0, type=int, help="random seed")
    parser.add_argument("-e", "--epoch", default=200, type=int, help="epochs to train and val")
    parser.add_argument("-l", "--lr", default=1e-4, type=float, help="learning rate for optimizer")
    
    parser.add_argument("-cls", "--num_class", default=1, type=int, help="num of classes")
    parser.add_argument("-fd", "--features_dim", default=1000, type=int, help="DINOv2 features dim")
    parser.add_argument("-nb", "--num_block", default=3, type=int, help="num of BMSS blocks")
    parser.add_argument("-g", "--com_factor", default=64, type=int, help="compression factor G")

    args = parser.parse_args()

    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

    print("experiment name : {}".format(args.exp))
    print("num of epochs   : {:6d}".format(args.epoch))
    print("num of workers  : {:6d}".format(args.work))
    print("learning rate   : {:4f}".format(args.lr))
    print("device          : {}".format(device))
    print("seed            : {}".format(args.seed))

    # Initialize seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    logger = CompleteLogger("./exp_log/{}/{}".format(args.lr, args.exp))
    main(args)

    print("Done")
    logger.close()
