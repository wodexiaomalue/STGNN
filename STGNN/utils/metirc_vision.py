import pandas
import numpy as np
import matplotlib.pyplot as plt
#from setuptools.sandbox import save_path
import seaborn as sns
from sklearn.manifold import TSNE

predict_value_inverse = np.load(".npy")
test_gt = np.load(".npy")
stgnn_pre = np.load(".npy")

seqlen = 40
pred_lenth = 30
# mae=MAE(predict_value_inverse[:,-pred_lenth:,:], test_gt[:,-pred_lenth:,:] )
# rmse=RMSE(predict_value_inverse[:,-pred_lenth:,:], test_gt[:,-pred_lenth:,:])
# print(f"mae:{mae}, rmse:{rmse}")
# 可视化
#----------------------------------------------------------------------------
# fig, axes = plt.subplots(3, 3, figsize=(14, 9))
# savepath = './' + f'/distribution/InScatter2_{seqlen}_{pred_lenth}.png'
# for ax, station in zip(axes.ravel(), range(70,80)):
#     all_data = np.vstack([test_gt[:,-pred_lenth:,station], predict_value_inverse[:,-pred_lenth:,station], stgnn_pre[:,:,station]])
#     labels = np.array([0] * len(test_gt) + [1] * len(predict_value_inverse) + [2] * len(stgnn_pre))
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     data_2d = tsne.fit_transform(all_data)
#
#     ax.scatter(data_2d[labels == 0, 0], data_2d[labels == 0, 1], c='blue', label='Original Data', alpha=0.5)
#
#     # 合成数据点（橙色）
#     #ax.scatter(data_2d[labels == 1, 0], data_2d[labels == 1, 1], c='red', label='DSTGNN', alpha=0.5)
#
#     ax.scatter(data_2d[labels == 2, 0], data_2d[labels == 2, 1], c='orange', label='STGNN', alpha=0.5)
#
#     ax.legend()
#
# plt.tight_layout()
# plt.savefig(savepath, dpi=300, bbox_inches='tight')
# plt.show()
# ----------------------------------------------------------------------------------
# plt.figure(figsize=(12, 8)
# index = 4

station = 73
for index in range(10, 20):
    # savepath= './' + f'/visual{pred_lenth}/In_{seqlen}_{pred_lenth}_{station}_{index}.png'

    plt.figure(figsize=(6, 6))
    # lower_quantile1 = np.percentile(test_gt, 5, axis=0)
    # upper_quantile1 = np.percentile(test_gt, 95, axis=0)
    # lower_quantile2 = np.percentile(stgnn_pre[index], 5, axis=0)
    # upper_quantile2 = np.percentile(stgnn_pre[index], 95, axis=0)

    pre = predict_value_inverse[index,:,station ]
    tre = test_gt[index,:,station ]
    stgnnpre = stgnn_pre[index,:,station]

    # -----------------------------------------------------------------------------------------------------

    # a=np.array([pre[-pred_lenth-1]])
    # stgnnpre = np.concatenate((np.array([pre[-pred_lenth-1]]), stgnnpre))
    # # 绘制 tre 的前 14 个数据 (history)
    # plt.plot(range(seqlen), tre, label="History", color='blue', linestyle='-')
    #
    # # 绘制 tre 的后 10 个数据
    # plt.plot(range(seqlen - pred_lenth, seqlen), tre[-pred_lenth:], label="Ground Truth", color='green', linestyle='-')
    #
    # # 绘制 pre 的后 10 个数据
    # plt.plot(range(seqlen - pred_lenth-1, seqlen), pre[-pred_lenth-1:], label="DSTGNN", color='red', linestyle='--', marker='x')
    # #plt.fill_between(range(seqlen - pred_lenth-1, seqlen), lower_quantile1[-pred_lenth-1:, station], upper_quantile1[-pred_lenth-1:,station], color='grey', alpha=0.3)
    #
    # plt.plot(range(seqlen - pred_lenth-1, seqlen), stgnnpre, label="STGNN", color='black', linestyle='--',marker='o')
    #
    # # 设置图例和标签
    # plt.xlabel("Time Step")
    # plt.ylabel("Value")
    # # plt.title("History, Ground Truth, and Prediction (Last 10 Steps)")
    # plt.xlim(0, seqlen)  # 设置x轴范围
    # plt.legend()
    # plt.tight_layout()
    # # plt.axis('equal')  # 保持坐标轴比例相等
    # plt.savefig(savepath, dpi=300, bbox_inches='tight')  # 保存为 PNG 格式，dpi 设置分辨率，bbox_inches='tight' 去除多余空白
    #
    # plt.show()

    # -----------------------------------------------------------------------------------------------------
    savepath =  './' + f'/distribution/Indistribution_{seqlen}_{pred_lenth}_{station}_{index}.png'
    sns.kdeplot(tre[-pred_lenth:],  color="blue", label="Original", linewidth=1.5)
    # 绘制生成数据分布（黄色虚线）
    sns.kdeplot(pre[-pred_lenth:], color="green", linestyle="--", label="DSTGNN", linewidth=1.5)

    sns.kdeplot(stgnnpre, color="orange", linestyle="--", label="STGNN", linewidth=1.5)


    # 设置图例和标签
    plt.xlabel("Data Value")
    plt.ylabel("Data Density Estimate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

