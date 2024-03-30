from sklearn.cluster import KMeans
import numpy as np


def k_center(time_series):

    # 应用K-Means算法进行聚类
    k = 10  # 聚类中心数量
    ts = []
    for i in range(len(time_series)):
        ts.append(time_series[i][1:])
    time_series=ts
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(time_series)

    # 获取每个时间序列所属的聚类中心标签
    labels = kmeans.labels_

    # 按照类别对时间序列进行排序
    sorted_time_series = [[] for _ in range(k)]
    for i, ts in enumerate(time_series):
        sorted_time_series[labels[i]].append(ts)

    # 计算最长类别的时间序列数量
    max_count = max(len(ts_list) for ts_list in sorted_time_series)

    # 轮流排序不同类的时间序列
    merged_time_series = []
    for i in range(max_count):
        for ts_list in sorted_time_series:
            if i < len(ts_list):
                merged_time_series.append(ts_list[i])

    return merged_time_series

def herding(time_series):
    k = 500
    representative_samples = [time_series[0]]

    # 迭代选择代表性样本
    while len(representative_samples) < k:
        distances = np.zeros(len(time_series))
        for i, ts in enumerate(time_series):
            # 计算样本ts与已选择的代表性样本之间的距离（例如欧氏距离）
            distances[i] = np.linalg.norm(ts - representative_samples, axis=1).min()
        max_distance_idx = np.argmax(distances)
        representative_samples.append(time_series[max_distance_idx])


    for sample in representative_samples:
        print(sample)
    return representative_samples


def gradient_matching(time_series):
    reference_ts = time_series[0]

    reference_gradient = np.gradient(reference_ts)
    gradient_diffs = []
    for ts in time_series:
        current_gradient = np.gradient(ts)
        diff = np.linalg.norm(reference_gradient - current_gradient)
        gradient_diffs.append(diff)

    k = 500
    kmeans = KMeans(n_clusters=k)
    X = np.array(gradient_diffs).reshape(-1, 1)
    kmeans.fit(X)
    representative_indices = np.argpartition(kmeans.cluster_centers_.flatten(), k)[:k]
    representative_samples = time_series[representative_indices]
    return representative_samples