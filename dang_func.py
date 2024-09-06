import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import learning_curve, train_test_split

# 데이터 분할 함수
def split_and_convert_data(X, y, test_size=0.2, random_state=42):
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_test = X_test.values
    y_test = y_test.values.ravel()  # y_test를 1차원 배열로 변환

    return X_train, X_test, y_train, y_test

# 성능, 혼동행렬 함수
def get_clf_eval(y_test, pred, pred_proba):
    # 정확도, 정밀도, 재현율, F1 스코어 계산
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    # ROC AUC 스코어 계산
    roc_auc = None
    if pred_proba is not None:
        roc_auc = roc_auc_score(y_test, pred_proba)
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, pred)
    
    # 결과 출력
    st.subheader("모델 평가 지표")
    st.write(f"**정확도**: {accuracy:.2f}")
    st.write(f"**정밀도**: {precision:.2f}")
    st.write(f"**재현율**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")
    if roc_auc is not None:
        st.write(f"**ROC AUC Score**: {roc_auc:.2f}")
    
    # 혼동 행렬 시각화
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    plt.close()  # 메모리 누수 방지를 위해 플롯 닫기

# ROC 커브 함수
def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    plt.close()


# 러닝커브 함수
def plot_learning_curve_accuracy(estimator, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring='accuracy'  # 정확도를 사용하기 위해 accuracy를 선택
    )

    # 평균 및 표준편차 계산
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 그래프 그리기
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training accuracy')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation accuracy')

    # 표준편차로 영역 표시
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve (Accuracy)')
    plt.legend(loc='best')
    plt.grid()

    # Streamlit에 그래프 표시
    st.pyplot(plt)
    plt.close()  # plt.close()를 추가하여 메모리 누수를 방지

#페이지 이동 함수
def go_to_page(page_name):
    st.session_state.page = page_name

def remove_outliers(df, threshold=3):
    df_filtered = df.copy()  # 원본 데이터프레임 복사
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mean = df[column].mean()
            std_dev = df[column].std()
            z_scores = (df[column] - mean) / std_dev
            df_filtered = df_filtered[np.abs(z_scores) <= threshold]
    return df_filtered