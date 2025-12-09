import torch
from prepareData import prepare_data
import numpy as np
from torch import optim
from param import parameter_parser
from model import FKAMHV
from utils import get_L2reg, Myloss,kl_loss
from traindata import Dataset
import ConstructHW
import torch.nn.functional as F
from Calculate_Metrics import Metric_fun
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import pandas as pd



warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#traindata要改回来，preparedata要改回来


def train_epoch(model, train_data, optim, opt):
    model.train()

    regression_crit = Myloss()

    one_index = train_data[8][0].to(device).t().tolist()
    zero_index = train_data[8][1].to(device).t().tolist()

    dis_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)
    A = train_data[6].to(device)
    AT=A.T
    AT = AT.to(device)



    concat_miRNA = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA)
    concat_mi_tensor = concat_mi_tensor.to(device)

    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[3], is_probH=False)
    G_mi_Kn = G_mi_Kn.to(device)


    concat_dis = np.hstack(
        [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)

    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[3], is_probH=False)
    G_dis_Kn = G_dis_Kn.to(device)


    loss_kl = kl_loss(788, 374)
    #loss_kl = kl_loss(495, 380)       #main函数里model的初始参数，model自身的初始参数，param参数初始改一下，hyperedge_encoder和node_encoder里的加数改一下。


    for epoch in range(1, opt.epoch + 1):


        score, mi_cl_loss, dis_cl_loss,reconstruction_en = model(concat_mi_tensor, concat_dis_tensor,  G_mi_Kn, G_dis_Kn, AT, A)
        reconstruction_en=reconstruction_en.to(torch.float32).to(device)
        reconstruction_en = torch.sigmoid(reconstruction_en)
        BCE = torch.nn.functional.binary_cross_entropy(reconstruction_en, A, reduction='mean')
        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)
        loss_beta =BCE + 0.5*loss_k




        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)#double
        reg_loss = get_L2reg(model.parameters())#float
        recover_loss=recover_loss.to(torch.float32)
        tol_loss = (recover_loss + 0.9*(2 * mi_cl_loss + 2 * dis_cl_loss) + 0.00001 * reg_loss + 0.1*loss_beta)


        optim.zero_grad()
        tol_loss.backward()
        optim.step()
        # # 计算训练损失
        # train_loss = tol_loss.item()  # 获取标量值
        # print(f'Epoch {epoch}/{opt.epoch}, Train Loss: {train_loss:.4f}')


    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor,
                                                                          concat_dis_tensor,
                                                                          G_mi_Kn, G_dis_Kn, AT,A)



    return true_value_one, true_value_zero, pre_value_one, pre_value_zero






def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,AT,A):
    model.eval()
    score, _, _ , _= model(concat_mi_tensor, concat_dis_tensor,
                        G_mi_Kn, G_dis_Kn, AT, A)

    test_one_index = data[9][0].t().tolist()
    test_zero_index = data[9][1].t().tolist()
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]




    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]
    return true_one, true_zero, pre_one, pre_zero


def evaluate(true_one, true_zero, pre_one, pre_zero):
    Metric = Metric_fun()

    test_po_num = true_one.shape[0]
    test_index = np.array(np.where(true_zero == 0))
    np.random.shuffle(test_index.T)
    test_ne_index = tuple(test_index[:, :test_po_num])

    eval_true_zero = true_zero[test_ne_index]
    eval_true_data = torch.cat([true_one, eval_true_zero])

    eval_pre_zero = pre_zero[test_ne_index]
    eval_pre_data = torch.cat([pre_one, eval_pre_zero])

    # 对 eval_pre_data 的值进行排序，并获取相应的索引
    sorted_eval_pre_data, sorted_indices = torch.sort(eval_pre_data, descending=True)

    #metrics_tensor = Metric.cv_mat_model_evaluate1(eval_true_data, eval_pre_data)


    return sorted_eval_pre_data, sorted_indices


def main(opt):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)




    hidden_list = [256, 256]
    num_proj_hidden = 64

    model = FKAMHV(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args, num_in_node=374, num_in_edge=788,
                       num_hidden1=512, num_out=64)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(model, train_data[0], optimizer, opt)
    sorted_eval_pre_data, sorted_indices = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)

    print('sorted_eval_pre_data:', sorted_eval_pre_data)
    print('sorted_indices:', sorted_indices)

    print('sorted_eval_pre_data.shape:', sorted_eval_pre_data.shape)
    print('sorted_indices:', sorted_indices.shape)

    # 如果 sorted_eval_pre_data 是一个 Tensor，且需要从计算图中分离出来
    sorted_eval_pre_data = sorted_eval_pre_data.detach().cpu().numpy() if isinstance(sorted_eval_pre_data,
                                                                                     torch.Tensor) else sorted_eval_pre_data

    # 进行最大最小归一化，将值范围压缩到 0 到 1 之间
    min_value = np.min(sorted_eval_pre_data)
    max_value = np.max(sorted_eval_pre_data)

    # 最大最小归一化
    normalized_values = (sorted_eval_pre_data - min_value) / (max_value - min_value)

    # 将每个索引值加 1
    sorted_indices_plus_one = [index + 1 for index in sorted_indices.tolist()]

    # 创建DataFrame，包含加1后的排序索引和归一化后的预测值
    df = pd.DataFrame({
        'Sorted Indices (plus 1)': sorted_indices_plus_one,
        'Normalized Predicted Values': normalized_values.tolist()
    })

    # 将DataFrame保存到Excel文件
    output_excel_path = 'sorted_predictions_normalized.xlsx'  # 文件保存路径
    df.to_excel(output_excel_path, index=False)

    print(f"Sorted predictions with normalization and indices (with +1) have been saved to {output_excel_path}")






if __name__ == '__main__':
    args = parameter_parser()
    main(args)