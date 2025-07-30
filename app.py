import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("تنبؤ سعر العقار باستخدام Linear Regression")

uploaded_file = st.file_uploader("ارفع ملف train (1).csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("عرض البيانات:")
    st.dataframe(df.head())

    features = ['GrLivArea', 'BedroomAbvGr', 'OverallQual', 'Neighborhood']
    target = 'SalePrice'

    df = df[features + [target]].dropna()
    le = LabelEncoder()
    df['Neighborhood'] = le.fit_transform(df['Neighborhood'])

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    st.subheader(f"دقة النموذج: {accuracy:.2f}")

    st.subheader("علاقة المساحة بسعر العقار")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], color='blue', ax=ax)
    ax.set_title('علاقة المساحة بسعر العقار', fontsize=14)
    ax.set_xlabel('المساحة (قدم مربع)', fontsize=12)
    ax.set_ylabel('السعر ($)', fontsize=12)
    st.pyplot(fig)

    st.subheader("جرب التنبؤ بسعر عقار جديد:")
    area = st.number_input("المساحة (GrLivArea)", value=1500)
    bedrooms = st.number_input("عدد غرف النوم (BedroomAbvGr)", value=3)
    quality = st.slider("التقييم العام (OverallQual)", 1, 10, 5)
    neighborhood = st.selectbox("الحي (Neighborhood)", df['Neighborhood'].unique())

    user_input = pd.DataFrame({
        'GrLivArea': [area],
        'BedroomAbvGr': [bedrooms],
        'OverallQual': [quality],
        'Neighborhood': [neighborhood]
    })

    prediction = model.predict(user_input)[0]
    st.success(f"سعر العقار المتوقع: ${prediction:,.2f}")
else:
    st.info("الرجاء رفع ملف بصيغة CSV يحتوي على البيانات.")