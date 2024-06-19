import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터 로드
ratings_path = 'ratings.dat'
ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python',
                      names=['User', 'Item', 'Rating', 'Timestamp'])

# user_item_matrix 생성
num_users = ratings['User'].nunique()
num_items = ratings['Item'].max()
user_item_matrix = np.full((num_users, num_items), np.nan)

# user_item_matrix의 값 입력
for row in ratings.itertuples():
    user_item_matrix[row.User - 1, row.Item - 1] = row.Rating

# Clustring을 위해서 NaN 값을 0으로 변환
nan_mask = np.isnan(user_item_matrix)
user_item_matrix[nan_mask] = 0

# K-Means 클러스터링 적용(k = 3)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
user_clusters = kmeans.fit_predict(user_item_matrix)

# NaN값 복원
user_item_matrix[nan_mask] = np.nan

# 클러스터링된 그룹 정보를 ratings에 추가
ratings['Group'] = user_clusters[ratings['User'] - 1]

#Borda Count (BC)
def BC(group_matrix):
    nan_mask_BC = np.isnan(group_matrix)
    ranks = np.zeros_like(group_matrix, dtype=float)
    for i, row in enumerate(group_matrix):
        unique_vals, inverse_indices, counts = np.unique(row, return_inverse=True, return_counts=True)
        ranks_for_row = np.zeros_like(row, dtype=float)
        current_rank = 0
        for val_index, count in enumerate(counts):
            next_rank = current_rank + count
            avg_rank = (current_rank + next_rank - 1) / 2.0
            ranks_for_row[inverse_indices == val_index] = avg_rank
            current_rank = next_rank
        ranks[i] = ranks_for_row
    ranks[nan_mask_BC]=np.nan
    scores = np.nansum(ranks, axis=0)
    return scores

#Copeland Rule (CR)
def CR(group_matrix):
    pairwise_comp = np.zeros(group_matrix.shape[1])
    for i in range(group_matrix.shape[1]):
        for j in range(i + 1, group_matrix.shape[1]):
            comp = (group_matrix[:, i] > group_matrix[:, j]).sum() - (group_matrix[:, i] < group_matrix[:, j]).sum()
            pairwise_comp[i] += comp
            pairwise_comp[j] -= comp
    return pairwise_comp

# 각 그룹에 대해 알고리즘 적용
results = {}
for group in range(num_clusters):
    group_matrix = user_item_matrix[user_clusters == group]
    results[group] = {
        'AU': np.nansum(group_matrix, axis=0),
        'Avg': np.nanmean(group_matrix, axis=0),
        'SC': np.nansum(~np.isnan(group_matrix), axis=0),
        'AV': np.nansum(group_matrix >= 4, axis=0),
        'BC': BC(group_matrix),
        'CR': CR(group_matrix)
    }

# 각 그룹과 각 알고리즘의 상위 n(10)개 결과 출력
top_n = 10
top_recommendations = {group: {} for group in range(num_clusters)}
for group in range(num_clusters):
    print(f"Group {group + 1} Top 10 recommendations using by 6 Algorithms")
    for algorithm, scores in results[group].items():
        top_recommendations[group][algorithm] = np.argsort(scores)[-top_n:][::-1]
        print(f"Algorithm({algorithm}) : {top_recommendations[group][algorithm] + 1}")
    print("\n")
