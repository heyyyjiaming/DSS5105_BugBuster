import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
import plotly.express as px

def categorize_performance_by_cluster(cluster):
    if cluster == e_cluster_centers['Cluster'].idxmax():
        return 'Good'
    elif cluster == e_cluster_centers['Cluster'].idxmin():
        return 'Poor'
    else:
        return 'Average'

# 使用新的聚类结果为 Company 1 分配 Performance Category
company_e_data['Performance Category'] = company_e_data['Cluster'].apply(categorize_performance_by_cluster)


# Step 2: 使用线性回归公式计算每年的 ESG 得分
# 假设 reg.coef_ 已经存储了模型的权重 (weights)，并且 intercept_b 已经存储了截距 (b)
e_weights = reg.coef_
e_intercept_b = reg.intercept_

# 将权重转换为NumPy数组，以便于矩阵运算
e_weights = np.array(e_weights)

# 定义计算得分的函数
def calculate_score(features, weights, intercept):
    return np.dot(features, weights) + intercept

# 计算每年 ESG 得分并保存到新的 DataFrame 中
company_e_scores = e_company_scaled.apply(lambda row: calculate_score(row, e_weights, e_intercept_b), axis=1)
company_e_data['Calculated Score'] = company_e_scores