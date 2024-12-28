import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100개의 랜덤 X값 생성
y = 4 + 3 * X + np.random.randn(100, 1)  # 선형 방정식 y = 4 + 3x + noise

# plt.scatter(X, y, color='blue', alpha=0.5)
# plt.title('Scatter Plot of Data')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.grid(True)
# plt.show()

model = LinearRegression()
model.fit(X, y)

print('기울기 : ',model.coef_)
print('절편 : ',model.intercept_)

X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
plt.plot(X_new, y_pred, color='red', label='Prediction Line')  # 예측된 직선
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

test_X = np.array([[1.5], [3.0]])  # X=1.5, X=3.0
test_y_pred = model.predict(test_X)

print("X=1.5일 때 예측 값:", test_y_pred[0])
print("X=3.0일 때 예측 값:", test_y_pred[1])


# 모델 평가
from sklearn.metrics import mean_squared_error

# 전체 데이터에 대한 예측값 계산
y_pred_all = model.predict(X)

# MSE 계산
mse = mean_squared_error(y, y_pred_all)
print('\n모델 평가:')
print(f'Mean Squared Error (MSE): {mse:.4f}')

# ... existing imports ...
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 데이터를 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge 회귀 모델의 하이퍼파라미터 튜닝
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

# Lasso 회귀 모델의 하이퍼파라미터 튜닝
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
lasso = Lasso()
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

# 각 모델의 최적 파라미터와 성능 출력
print("\n=== 모델 튜닝 결과 ===")
print("Ridge 최적 alpha:", ridge_grid.best_params_)
print("Ridge 최소 MSE:", -ridge_grid.best_score_)
print("Lasso 최적 alpha:", lasso_grid.best_params_)
print("Lasso 최소 MSE:", -lasso_grid.best_score_)

# 최적화된 모델로 테스트 세트 예측
best_ridge = ridge_grid.best_estimator_
best_lasso = lasso_grid.best_estimator_

# 각 모델의 테스트 세트 성능 평가
models = {
    'Linear Regression': model,
    'Ridge': best_ridge,
    'Lasso': best_lasso
}

print("\n=== 테스트 세트 성능 비교 ===")
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# 시각화
# plt.figure(figsize=(12, 6))
# for name, model in models.items():
#     y_pred = model.predict(X_test)
#     plt.scatter(y_test, y_pred, label=name, alpha=0.5)

# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('실제값')
# plt.ylabel('예측값')
# plt.title('모델별 예측 비교')
# plt.legend()
# plt.show()