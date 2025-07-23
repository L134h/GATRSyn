import argparse
import os
import time
import torch
import torch.nn as nn
import pickle
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, \
    cohen_kappa_score, f1_score
import seaborn as sns
from pathlib import Path

from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import DNN
from model.utils import save_args, save_best_model, find_best_model
from const import SYNERGY_FILE, DRUG2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device)

time_str = str(datetime.now().strftime('%y%m%d%H%M'))
LOG_INTERVAL = 20


def step_batch(model, batch, loss_func, gpu_id=None, train=True):
    drug1_feats, drug2_feats, cell_feats, y_true = batch
    if gpu_id is not None:
        drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.cuda(gpu_id), drug2_feats.cuda(gpu_id), \
            cell_feats.cuda(gpu_id), y_true.cuda(gpu_id)
    if train:
        y_pred = model(drug1_feats, drug2_feats, cell_feats)
    else:
        yp1 = model(drug1_feats, drug2_feats, cell_feats)
        yp2 = model(drug2_feats, drug1_feats, cell_feats)
        y_pred = (yp1 + yp2) / 2
    loss = loss_func(y_pred, y_true.squeeze().long())
    return loss, y_pred


def train_epoch(model, loader, loss_func, optimizer, gpu_id=None):
    model.train()
    epoch_loss = 0
    num_batches = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss, _ = step_batch(model, batch, loss_func, gpu_id)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    return epoch_loss / num_batches if num_batches > 0 else 0


def evaluate(model, loader, loss_func, gpu_id=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        num_batches = 0
        y_true_list = []
        y_pred_prob_list = []
        y_pred_class_list = []
        for batch in loader:
            loss, y_pred = step_batch(model, batch, loss_func, gpu_id, train=False)
            y_pred_softmax = torch.softmax(y_pred, dim=1).cpu()
            y_pred_softmax_numpy = y_pred_softmax.detach().numpy()
            y_pred_class = np.argmax(y_pred_softmax_numpy, axis=1)
            epoch_loss += loss.item()
            num_batches += 1
            y_true_list.extend(batch[3].squeeze().cpu().numpy())
            y_pred_prob_list.extend(y_pred_softmax_numpy[:, 1])
            y_pred_class_list.extend(y_pred_class)

        y_true = np.array(y_true_list)
        y_pred_prob = np.array(y_pred_prob_list)
        y_pred_class = np.array(y_pred_class_list)

        # Calculate metrics
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_prob)
            pr_auc = average_precision_score(y_true, y_pred_prob)
        else:
            auc = 0.5
            pr_auc = 0.5

        acc = accuracy_score(y_true, y_pred_class)
        bacc = balanced_accuracy_score(y_true, y_pred_class)
        prec = precision_score(y_true, y_pred_class, zero_division=0)
        tpr = recall_score(y_true, y_pred_class, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred_class, zero_division=0)

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        return avg_loss, auc, pr_auc, acc, bacc, prec, tpr, kappa, f1, y_true, y_pred_prob, y_pred_class


def save_results_to_txt(filename, epoch, metrics):
    """Save evaluation metrics to text file in specified format"""
    with open(filename, 'a') as f:
        f.write(f"{epoch}\t{metrics[1]:.4f}\t{metrics[2]:.4f}\t{metrics[3]:.4f}\t{metrics[4]:.4f}\t"
                f"{metrics[5]:.4f}\t{metrics[6]:.4f}\t{metrics[7]:.4f}\t{metrics[8]:.4f}\n")


def plot_confusion_matrix(y_true, y_pred, classes, filename):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(args, model, optimizer, loss_func, train_loader, valid_loader, test_loader,
                n_epoch, patience, gpu_id, out_dir, suffix):
    # Create results file with header
    results_file = os.path.join(out_dir, f'results_{suffix}.txt')
    with open(results_file, 'w') as f:
        f.write("Epoch\tAUC\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tF1\n")

    best_auc = 0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(1, n_epoch + 1):
        if early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

        # Train
        start_time = time.time()
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
        train_time = time.time() - start_time

        # Validate
        start_time = time.time()
        val_loss, val_auc, *val_metrics = evaluate(model, valid_loader, loss_func, gpu_id)
        val_time = time.time() - start_time

        # Log progress
        logging.info(f"Epoch {epoch}/{n_epoch}: "
                     f"Train Loss = {trn_loss:.4f} | "
                     f"Val Loss = {val_loss:.4f} (AUC: {val_auc:.4f}) | "
                     f"Times - Train: {train_time:.1f}s, Val: {val_time:.1f}s")

        # Save best model based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0
            save_best_model(model.state_dict(), out_dir, epoch, keep=1)
            logging.info(f"New best model at epoch {epoch} with Val AUC: {val_auc:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve}/{patience} epochs")

            # Check early stopping condition
            if epochs_no_improve >= patience:
                early_stop = True
                logging.info(f"Early stopping triggered after {patience} epochs without improvement")

    return best_auc, best_epoch, early_stop


def evaluate_final(model, test_loader, loss_func, gpu_id, out_dir, suffix):
    # Load best model
    best_model_path = find_best_model(out_dir)
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"Loaded best model for final evaluation from {best_model_path}")
    else:
        logging.warning("No best model found for final evaluation")
        return None, None, None

    # Evaluate on test set
    start_time = time.time()
    test_loss, test_auc, test_pr_auc, test_acc, test_bacc, test_prec, test_tpr, test_kappa, test_f1, \
        y_true, y_pred_prob, y_pred_class = evaluate(model, test_loader, loss_func, gpu_id)
    test_time = time.time() - start_time

    # Save final results
    with open(os.path.join(out_dir, f'results_{suffix}.txt'), 'a') as f:
        f.write("\nFinal Test Results:\n")
        f.write(f"AUC: {test_auc:.4f}\n")
        f.write(f"PR AUC: {test_pr_auc:.4f}\n")
        f.write(f"ACC: {test_acc:.4f}\n")
        f.write(f"Balanced ACC: {test_bacc:.4f}\n")
        f.write(f"Precision: {test_prec:.4f}\n")
        f.write(f"Recall: {test_tpr:.4f}\n")
        f.write(f"Kappa: {test_kappa:.4f}\n")
        f.write(f"F1: {test_f1:.4f}\n")

    # Log final results
    logging.info("=" * 60)
    logging.info(f"FINAL TEST RESULTS:")
    logging.info(f"AUC: {test_auc:.4f} | PR AUC: {test_pr_auc:.4f} | ACC: {test_acc:.4f}")
    logging.info(f"Balanced ACC: {test_bacc:.4f} | Precision: {test_prec:.4f}")
    logging.info(f"Recall: {test_tpr:.4f} | Kappa: {test_kappa:.4f} | F1: {test_f1:.4f}")
    logging.info(f"Test time: {test_time:.1f}s")
    logging.info("=" * 60)

    # Save predictions
    preds_file = os.path.join(out_dir, f'predictions_{suffix}.csv')
    np.savetxt(preds_file, np.column_stack((y_true, y_pred_class, y_pred_prob)),
               delimiter=',', fmt=['%d', '%d', '%.4f'],
               header='TrueLabel,PredLabel,PredProbability', comments='')

    # Plot confusion matrix
    plot_file = os.path.join(out_dir, f'confusion_matrix_{suffix}.png')
    plot_confusion_matrix(y_true, y_pred_class, ['Synergistic', 'Antagonistic'], plot_file)

    return test_auc, test_pr_auc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500, help="n epoch")
    parser.add_argument('--batch', type=int, default=256, help="batch size")
    parser.add_argument('--gpu', type=int, default=0, help="cuda device")
    parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
    parser.add_argument('--hidden', type=int, nargs='+', default=[1024, 2048, 4096, 8192], help="hidden size(s)")
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4, 1e-5], help="learning rate(s)")
    parser.add_argument('--suffix', type=str, default=time_str, help="output suffix")
    args = parser.parse_args()

    # Fixed folds
    TRAIN_FOLD = [2]
    VALID_FOLD = [1]
    TEST_FOLD = [0]

    # Load datasets (outside the loop as they are fixed)
    logging.info("Loading datasets...")
    train_data = FastSynergyDataset(
        DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
        SYNERGY_FILE, use_folds=TRAIN_FOLD
    )
    valid_data = FastSynergyDataset(
        DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
        SYNERGY_FILE, use_folds=VALID_FOLD, train=False
    )
    test_data = FastSynergyDataset(
        DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
        SYNERGY_FILE, use_folds=TEST_FOLD, train=False
    )

    # Create data loaders
    train_loader = FastTensorDataLoader(
        *train_data.tensor_samples(), batch_size=args.batch, shuffle=True
    )
    valid_loader = FastTensorDataLoader(
        *valid_data.tensor_samples(), batch_size=args.batch
    )
    test_loader = FastTensorDataLoader(
        *test_data.tensor_samples(), batch_size=args.batch
    )

    # GPU setup
    gpu_id = args.gpu if torch.cuda.is_available() else None

    # Prepare for hyperparameter search
    best_val_auc = 0
    best_params = {}
    best_dir = ""
    best_test_metrics = {}

    # Create master output directory
    master_out_dir = os.path.join(OUTPUT_DIR, f"hparam_search_{args.suffix}")
    os.makedirs(master_out_dir, exist_ok=True)

    # Hyperparameter loop
    total_combinations = len(args.hidden) * len(args.lr)
    current_run = 1
    start_time_total = time.time()

    # Create summary file for all runs
    summary_file = os.path.join(master_out_dir, 'hyperparameter_summary.csv')
    with open(summary_file, 'w') as f:
        f.write("RunID,HiddenSize,LearningRate,BestEpoch,ValAUC,TestAUC,TestPRAUC,TestACC,EarlyStop\n")

    for hidden_size in args.hidden:
        for lr_val in args.lr:
            # Create unique run identifier
            run_id = f"hidden{hidden_size}_lr{lr_val}"
            run_time_str = datetime.now().strftime('%H%M%S')
            run_suffix = f"{args.suffix}_{run_time_str}_{run_id}"
            out_dir = os.path.join(master_out_dir, f"train_{run_suffix}")
            os.makedirs(out_dir, exist_ok=True)

            # Setup logging for this run
            log_file = os.path.join(out_dir, f'train_{run_suffix}.log')
            logging.basicConfig(
                filename=log_file,
                format='%(asctime)s %(message)s',
                datefmt='[%Y-%m-%d %H:%M:%S]',
                level=logging.INFO,
                force=True  # Force reset logger handlers
            )
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)

            # Log run info
            logging.info(f"Hyperparameter search run {current_run}/{total_combinations}")
            logging.info(f"  Hidden size: {hidden_size}")
            logging.info(f"  Learning rate: {lr_val}")
            logging.info(f"  Output directory: {out_dir}")

            # Save run-specific args
            run_args = argparse.Namespace(
                epoch=args.epoch,
                batch=args.batch,
                gpu=args.gpu,
                patience=args.patience,
                hidden=hidden_size,
                lr=lr_val,
                suffix=run_suffix
            )
            save_args(run_args, os.path.join(out_dir, 'args.json'))

            # Initialize model with current hyperparameters
            model = DNN(2 * train_data.cell_feat_len() + 2 * train_data.drug_feat_len(), hidden_size)
            if gpu_id is not None:
                model = model.cuda(gpu_id)

            # Loss and optimizer
            loss_func = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_val)

            # Train model with early stopping
            start_time = time.time()
            val_auc, best_epoch, early_stop = train_model(
                run_args, model, optimizer, loss_func,
                train_loader, valid_loader, test_loader,
                run_args.epoch, run_args.patience, gpu_id, out_dir, run_args.suffix
            )
            training_time = time.time() - start_time

            # Log training results
            logging.info(f"Training completed. Validation AUC: {val_auc:.4f} "
                         f"| Best epoch: {best_epoch} "
                         f"| Training time: {training_time:.1f}s")

            # Perform final test evaluation
            test_auc, test_pr_auc, test_acc = evaluate_final(model, test_loader, loss_func, gpu_id, out_dir,
                                                             run_args.suffix)

            # Record best parameters
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_params = {'hidden': hidden_size, 'lr': lr_val}
                best_dir = out_dir
                best_test_metrics = {
                    'test_auc': test_auc,
                    'test_pr_auc': test_pr_auc,
                    'test_acc': test_acc
                }
                logging.info(f"New best hyperparameters: hidden={hidden_size}, lr={lr_val}, Val AUC={val_auc:.4f}")

            # Save run summary to master file
            with open(summary_file, 'a') as f:
                f.write(f"{run_suffix},{hidden_size},{lr_val},{best_epoch},{val_auc:.4f},")
                if test_auc is not None:
                    f.write(f"{test_auc:.4f},{test_pr_auc:.4f},{test_acc:.4f},")
                else:
                    f.write("N/A,N/A,N/A,")
                f.write(f"{early_stop}\n")

            current_run += 1

    # Final report
    total_time = time.time() - start_time_total
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logging.info("\n\n" + "=" * 80)
    logging.info("HYPERPARAMETER SEARCH COMPLETED")
    logging.info(f"Total search time: {int(hours)}h {int(minutes)}min {int(seconds)}s")
    logging.info(f"Total runs: {total_combinations}")
    logging.info("Best parameters:")
    logging.info(f"  Hidden size: {best_params.get('hidden', 'N/A')}")
    logging.info(f"  Learning rate: {best_params.get('lr', 'N/A')}")
    logging.info(f"  Validation AUC: {best_val_auc:.4f}")
    if best_test_metrics:
        logging.info(f"  Test AUC: {best_test_metrics['test_auc']:.4f}")
        logging.info(f"  Test PR AUC: {best_test_metrics['test_pr_auc']:.4f}")
        logging.info(f"  Test ACC: {best_test_metrics['test_acc']:.4f}")
    logging.info(f"  Output directory: {best_dir}")
    logging.info("=" * 80)

    # Save best parameters to master directory
    best_params_file = os.path.join(master_out_dir, 'best_params.txt')
    with open(best_params_file, 'w') as f:
        f.write(f"Best validation AUC: {best_val_auc:.4f}\n")
        f.write(f"Hidden size: {best_params.get('hidden', 'N/A')}\n")
        f.write(f"Learning rate: {best_params.get('lr', 'N/A')}\n")
        if best_test_metrics:
            f.write(f"Test AUC: {best_test_metrics['test_auc']:.4f}\n")
            f.write(f"Test PR AUC: {best_test_metrics['test_pr_auc']:.4f}\n")
            f.write(f"Test ACC: {best_test_metrics['test_acc']:.4f}\n")
        f.write(f"Output directory: {best_dir}\n")


if __name__ == '__main__':
    main()