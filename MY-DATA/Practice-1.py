from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터셋 나누기 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(wine.feature_names)
# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트, SVM, KNN 모델 생성
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}


# 각 모델 학습 및 평가
for name, model in models.items():
    print(f"\n=== {name} 모델 평가 ===")
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가 결과 출력
    print(classification_report(y_test, y_pred))
    print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
