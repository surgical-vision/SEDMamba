import csv
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from logger import CompleteLogger
from dataload_rarp import CustomVideoDataset
from baseline.SEDMamba import MultiStageModel

# Inference function
def inference_model(args, data_split_test_path):
    model = MultiStageModel(args.num_block, args.com_factor, args.features_dim, args.num_class)
    model.to(device)
    
    # Load the best model weights
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = CustomVideoDataset(data_split_test_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.work
    )

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

    print("test AUC: {:4f}%".format(test_AUC * 100))
    print("test mAP: {:4f}%".format(test_mAP * 100))

    return test_all_preds, test_all_scores, test_all_labels, test_each_vidoe_names, test_video_lengths

# Main function
def main(args):
    root_data_path = args.data_path
    data_split_test_path = root_data_path + "/test_emb_DINOv2/"

    test_all_preds, test_all_scores, test_all_labels, test_each_vidoe_names, test_video_lengths = inference_model(args, data_split_test_path)

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

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED")
    parser.add_argument("-exp", default="Inference-SEDMamba", type=str, help="exp name")
    parser.add_argument("-dp", "--data_path", default="/path/to/your/data", type=str, help="path to data")
    parser.add_argument("-lr", "--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("-w", "--work", default=4, type=int, help="num of workers")
    parser.add_argument("-gpu_id", type=str, nargs="?", default="cuda:0", help="device id to run")
    parser.add_argument("-cls", "--num_class", default=1, type=int, help="num of classes")
    parser.add_argument("-fd", "--features_dim", default=1000, type=int, help="DINOv2 features dim")
    parser.add_argument("-nb", "--num_block", default=3, type=int, help="num of BMSS blocks")
    parser.add_argument("-g", "--com_factor", default=64, type=int, help="compression factor G")
    parser.add_argument("-weight", "--weight_path", default="/path/to/your/model.pth", type=str, help="path to the trained model")

    args = parser.parse_args()

    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

    print("experiment name : {}".format(args.exp))
    print("device          : {}".format(device))

    logger = CompleteLogger("./exp_log/{}/{}".format(args.lr, args.exp))
    main(args)

    print("Inference Done")
    logger.close()
