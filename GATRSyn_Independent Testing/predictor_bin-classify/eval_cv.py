import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import DNN
from model.utils import conf_inv, calc_stat, find_best_model
from const import SYNERGY_FILE, DRUG2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR

time_str = str(datetime.now().strftime('%y%m%d%H%M'))   #设置时间戳
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    #调用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device)

def create_model(data, hidden_size, gpu_id=None):
    model = DNN(2 * data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    # model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model


def calc_metrics(out_dir):
    n_folds = 5
    all_y_preds = []
    all_y_trues = []
    all_drugA_names = []
    all_drugB_names = []
    all_cell_names = []

    for test_fold in range(n_folds):
        test_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
                                       SYNERGY_FILE, use_folds=[test_fold], train=False)
        test_mdl_dir = os.path.join(out_dir, str(test_fold))
        try:
            model = create_model(test_data, 4096, None)
            model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
        except Exception:
            try:
                model = create_model(test_data, 8192, None)
                model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
            except Exception:
                model = create_model(test_data, 2048, None)
                model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
        test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))
        model.eval()
        with torch.no_grad():
            for (drug1_feats, drug2_feats, cell_feats, y_true, drug1_ids, drug2_ids, cell_ids) in test_loader:
                yp1 = model(drug1_feats, drug2_feats, cell_feats)
                yp2 = model(drug2_feats, drug1_feats, cell_feats)
                y_pred = (yp1 + yp2) / 2
                y_pred_softmax = torch.softmax(y_pred, dim=1).cpu()
                y_pred_softmax_numpy = y_pred_softmax.detach().numpy()
                y_pred = np.argmax(y_pred_softmax_numpy, axis=1)
                y_true = y_true.numpy().flatten()

                all_y_preds.extend(y_pred)
                all_y_trues.extend(y_true)
                all_drugA_names.extend(drug1_ids)
                all_drugB_names.extend(drug2_ids)
                all_cell_names.extend(cell_ids)

                # 创建 DataFrame 并保存到 CSV 文件
    df = pd.DataFrame({
        'drugA': all_drugA_names,
        'drugB': all_drugB_names,
        'cell': all_cell_names,
        'y_pred': all_y_preds
    })

    cf = pd.DataFrame({
        'drugA': all_drugA_names,
        'drugB': all_drugB_names,
        'cell': all_cell_names,
        'y_true': all_y_trues
    })

    output_path = r"D:\LSL\MyFILES\PRO_2023attensyn\NEW_OnePPI(drug)_sim\predictor_bin-classify\output\cv_2410161001_BA"
    df.to_csv(os.path.join(output_path, "all_folds_y_pre.csv"), index=False)
    cf.to_csv(os.path.join(output_path, "all_folds_y_true.csv"), index=False)
    #         for (drug1_feats, drug2_feats, cell_feats, y_true, drug1_ids, drug2_ids, cell_ids) in test_loader:
    #             yp1 = model(drug1_feats, drug2_feats, cell_feats)
    #             yp2 = model(drug2_feats, drug1_feats, cell_feats)
    #             y_pred = (yp1 + yp2) / 2
    #             y_pred_softmax = torch.softmax(y_pred, dim=1).cpu()
    #             y_pred_softmax_numpy = y_pred_softmax.detach().numpy()
    #             y_pred = np.argmax(y_pred_softmax_numpy, axis=1)
    #             y_true = y_true.numpy().flatten()
    #
    #             y_preds.extend(y_pred)
    #             y_trues.extend(y_true)
    #             drugA_names.extend(drug1_ids)
    #             drugB_names.extend(drug2_ids)
    #             cell_names.extend(cell_ids)
    #
    #         # 创建 DataFrame 并保存到 CSV 文件
    # df = pd.DataFrame({
    #     'drugA': drugA_names,
    #     'drugB': drugB_names,
    #     'cell': cell_names,
    #     'y_pred': y_preds
    # })
    # cf = pd.DataFrame({
    #     'drugA': drugA_names,
    #     'drugB': drugB_names,
    #     'cell': cell_names,
    #     'y_true': y_trues
    # })
    #
    # df.to_csv(os.path.join(r"D:\LSL\MyFILES\PRO_2023attensyn\NEW_OnePPI(drug)_sim\predictor_bin-classify\output\cv_2409231923_AB\y_pre", str(test_fold) + ".csv"), index=False)
    # cf.to_csv(os.path.join(r"D:\LSL\MyFILES\PRO_2023attensyn\NEW_OnePPI(drug)_sim\predictor_bin-classify\output\cv_2409231923_AB\y_true", str(test_fold) + ".csv"), index=False)

    # with open(os.path.join(out_dir, 'y_preds.pkl'), 'wb') as f:
    #     pickle.dump(y_preds, f)
    # with open(os.path.join(out_dir, 'y_trues.pkl'), 'wb') as f:
    #     pickle.dump(y_trues, f)

#输出各折的MSE和PEARSON，只需为以下mdl_dir增加输入
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mdl_dir', type=str, nargs='?',  default=r'D:\LSL\MyFILES\PRO_2023attensyn\NEW_OnePPI(drug)_sim\predictor_bin-classify\output\cv_2410161001_BA', help="model dir")
    args = parser.parse_args()
    mdl_dir = os.path.join(OUTPUT_DIR, args.mdl_dir)
    calc_metrics(mdl_dir)  ##保存计算指标结果到路径中


if __name__ == '__main__':
    main()
