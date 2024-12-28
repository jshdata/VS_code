import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 샘플 데이터 생성
np.random.seed(42)
data, labels = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 데이터 시각화
plt.scatter(data[:, 0], data[:, 1], s=30, color='gray')
plt.title("Generated Data")
plt.show()

# K-means 알고리즘 적용
kmeans = KMeans(n_clusters=4, random_state=42)  # 클러스터 개수 4개로 설정
kmeans.fit(data)

# 클러스터 할당 및 중심 추출
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 결과 시각화
for i in range(4):
    cluster_points = data[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")
    
    # 각 군집의 중심에서 가장 먼 점까지의 거리 계산
    distances = np.sqrt(np.sum((cluster_points - centroids[i])**2, axis=1))
    radius = np.max(distances)
    
    # 원 그리기
    circle = plt.Circle(centroids[i], radius, fill=False, color='black', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label="Centroids")

plt.legend()
plt.title("K-means Clustering")
plt.gca().set_aspect('equal')  # 축 비율을 1:1로 설정
plt.show()

