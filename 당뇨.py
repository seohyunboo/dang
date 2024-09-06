## 기본적인 툴 각각 탭이나 사이드바를 맡아 시각화하여 표현하기
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import learning_curve, train_test_split
from dang_func import split_and_convert_data, get_clf_eval, plot_roc_curve, plot_learning_curve_accuracy, go_to_page

# 모델 불러오기
model_total = joblib.load("data/pred.pkl")
model_nonsampling = joblib.load("data/pred_nonsampling.pkl")

# 페이지 설정
st.set_page_config(layout="wide")

# 여백과 콘텐츠를 비율로 나눔
empty1, con1, empty2 = st.columns([0.1, 1.0, 0.2])

with st.sidebar: # 사이드바 내에서 아래 코드를 실행
       with st.sidebar: # 사이드바 내에서 아래 코드를 실행
       # radio 타이틀 설정
        st.title("Choose a plotting library")
        # radio 설정 및 생성
        plot_type = st.radio(
            "당신의 성별은?",
            ("남자", "여자"))

        # 각각의 값에 해당하는 그래프 표시되게 하기
        if plot_type == "남자":
            sex = 1
        elif plot_type == "여자":
            sex = 2

        # 슬라이더 추가 생성 후 제목과 범위(0~100)지정
        age = st.sidebar.text_input("당신의 나이", value="0")
        HE_chol = st.sidebar.text_input("총 콜레스테롤 수치", value="0")
        HE_wc = st.sidebar.text_input("허리 둘레", value="0")
        HE_crea = st.sidebar.text_input("크레아티닌 수치", value="0")
        HE_alt = st.sidebar.text_input("alt 수치(간수치)", value="0")
        HE_TG = st.sidebar.text_input("중성 지방 수치", value="0")
        HE_Upro = st.sidebar.text_input("요단백 수치", value="0")

        # 입력 값을 적절한 타입으로 변환
        sex = float(sex)
        age = float(age)
        HE_chol = float(HE_chol)
        HE_wc = float(HE_wc)
        HE_crea = float(HE_crea)
        HE_alt = float(HE_alt)
        HE_TG = float(HE_TG)
        HE_Upro = float(HE_Upro)
        # 관련 링크 사이드바에 추가
        st.sidebar.markdown('Tickers Link : [건강 관리 협회](https://www.kahp.or.kr/home/kahp_kor/hlthChk/hlthChkPgm/ntnHlthChk.jsp)')

        st.write(f"""
        입력한 정보 (확인해주세요):
        - 성별: {sex}
        - 나이: {age}
        - 총 콜레스테롤: {HE_chol}
        - 허리 둘레: {HE_wc}
        - 총 크레아티닌: {HE_crea}
        - 총 ALT: {HE_alt}
        - 중성 지방: {HE_TG}
        - 총 요단백: {HE_Upro}
        """)

## main page ##
with con1:
    st.header("당뇨병 예측 모델")
    tab1, tab2, tab3, tab4 = st.tabs(["당뇨 현황", "모델 소개","사용자의 상태", "식단,운동추천"])

with tab1: # 전체 금액 히스토플랏 그리기
    fig, ax = plt.subplots()
    st.header("2019~2021년도 당뇨 현황")

with tab2:  
    # CSV 파일에서 데이터 로드
    X = pd.read_csv('data/X.csv')
    y = pd.read_csv('data/y.csv')
    X_nonsampling = pd.read_csv('data/X_nonsampling.csv')
    y_nonsampling = pd.read_csv('data/y_nonsampling.csv')

    # 페이지 상태 관리
    if 'page' not in st.session_state:
        st.session_state.page = 'intro'

    # 소개 페이지
    if st.session_state.page == 'intro':
        st.title("예측 모델 소개")
        st.session_state.model_choice = st.selectbox("모델 선택", ["최종 모델", "샘플링 안한 xgboost"])
        
        if st.button("성능 보기"):
            go_to_page('next')
    

    # 모델 성능 페이지
    elif st.session_state.page == 'next':
        if st.session_state.model_choice == "최종 모델":
            with st.spinner("최종 모델의 성능을 계산하는 중..."):
                model = model_total
                st.title("최종 모델 성능")
                X_train, X_test, y_train, y_test = split_and_convert_data(X, y, test_size=0.2, random_state=42)
                
                # 예측
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                # 성능, 혼동행렬
                get_clf_eval(y_test, y_pred, y_proba)

                # ROC Curve
                st.subheader("ROC Curve")
                plot_roc_curve(y_test, y_proba)



                # 러닝 커브 시각화
                st.subheader("Learning Curve")
                plot_learning_curve_accuracy(model, X_train, y_train)

        if st.session_state.model_choice == "샘플링 안한 xgboost":
            with st.spinner("샘플링 안한 xgboost 모델의 성능을 계산하는 중..."):
                model = model_nonsampling
                st.title("샘플링 안한 XGBoost 모델 성능")
                X_train, X_test, y_train, y_test = split_and_convert_data(X_nonsampling, y_nonsampling, test_size=0.2, random_state=42)
                
                # 예측
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                # 성능, 혼동행렬
                get_clf_eval(y_test, y_pred, y_proba)

                # ROC Curve
                st.subheader("ROC Curve")
                plot_roc_curve(y_test, y_proba)

                # 러닝 커브 시각화
                st.subheader("Learning Curve")
                plot_learning_curve_accuracy(model, X_train, y_train)

        #다시 소개 페이지로 돌아가는 버튼
        if st.button("돌아가기"):
            go_to_page('intro')
            
    

with tab3: # 사이드 바에서 입력 된 정보로 모델 돌리기
    st.title("XGBoost 모델 예측")

    input_data = input_data=np.array ([[HE_chol, HE_wc,	HE_crea, HE_alt, HE_TG,	age, sex, HE_Upro]])

    # 예측 수행
    prediction = model_total.predict(input_data)
    prediction_proba = model_total.predict_proba(input_data)[:, 1]  # 확률 추정

    # 결과 출력
    st.subheader("Prediction Results")
    st.write(f"Predicted class: {prediction[0]}")
    st.write(f"Probability of class 1: {prediction_proba[0]:.4f}")

with tab4: # 남,여 금액 비율 박스플랏 그리기
    fig, ax = plt.subplots()
    st.header("식단,운동추천")