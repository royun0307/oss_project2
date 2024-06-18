import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터 로드
ratings_path = 'ratings.dat'
ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python',
                      names=['User', 'Item', 'Rating', 'Timestamp'])

# Item를 연속된 범위로 매핑
unique_item = ratings['Item'].unique()
item_map = {item: idx for idx, item in enumerate(unique_item)}

# Item 컬럼을 매핑된 ID로 업데이트
ratings['Item'] = ratings['Item'].map(item_map)

# 사용자-아이템 매트릭스 생성
num_users = ratings['User'].nunique()
num_items = len(unique_item)
user_item_matrix = np.full((num_users, num_items), np.nan)

# 사용자-아이템 매트릭스 값 입력
for row in ratings.itertuples():
    user_item_matrix[row.User - 1, row.Item] = row.Rating

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

#Additive Utilitarian (AU)
def AU(group_matrix):
    return np.nansum(group_matrix, axis=0)

#Average (Avg)
def Avg(group_matrix):
    return np.nanmean(group_matrix, axis=0)

#Simple Count (SC)
def SC(group_matrix):
    return np.nansum(~np.isnan(group_matrix), axis=0)

#Approval Voting (AV) (default threshold = 4)
def AV(group_matrix, threshold = 4):
     return np.nansum(group_matrix >= threshold, axis=0)

#Borda Count (BC)
def BC(group_matrix):
    scores = np.argsort(np.argsort(group_matrix, axis=1), axis=1).sum(axis=0)
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
        'AU': AU(group_matrix),
        'Avg': Avg(group_matrix),
        'SC': SC(group_matrix),
        'AV': AV(group_matrix),
        'BC': BC(group_matrix),
        'CR': CR(group_matrix)
    }

# 각 그룹과 각 알고리즘의 상위 n(10)개 결과 얻기
top_n = 10
top_recommendations = {group: {} for group in range(num_clusters)}
for group in range(num_clusters):
    for algorithm, scores in results[group].items():
        top_recommendations[group][algorithm] = np.argsort(scores)[-top_n:][::-1]

# 각 그룹과 각 알고리즘의 상위 n(10)개 출력
for group in range(num_clusters):
    print(f"Group {group + 1} Top 10 recommendations using by 6 Algorithms")
    for algorithm in results[group].keys():
        print(f"Algorithm({algorithm}) : {top_recommendations[group][algorithm]}")
    print("\n")