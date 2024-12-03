import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

# Load dataset
data = pd.read_csv("Fish.csv")

# Sidebar with improved style
st.sidebar.image("ikan.png", width=150)  # Sesuaikan ukuran (contoh: 150 piksel)
st.sidebar.markdown("## **🐟 Navigasi**")
menu = st.sidebar.radio("Pilih Menu", ["🏠 Home", "📊 Visualisasi", "🤖 Model Prediksi"])

# Home page
if menu == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: blue;'>Prediksi Berat Ikan Mbah Tadji🎣</h1>", unsafe_allow_html=True)
    st.markdown("""Selamat datang! Aplikasi ini membantu Anda memprediksi berat ikan berdasarkan data pengukuran fisik. 🚀""")
    st.image("banner.jpg", use_container_width=True)
    
    # Display dataset
    with st.expander("📂 Lihat Dataset"):
        st.dataframe(data)

    # Display descriptive statistics
    if st.checkbox("📊 Tampilkan Statistik Deskriptif"):
        st.markdown("### Statistik Deskriptif")
        st.write(data.describe())

# Visualizations
elif menu == "📊 Visualisasi":
    st.markdown("<h1 style='color: green;'>Visualisasi Data Ikan</h1>", unsafe_allow_html=True)
    
    # Scatter plot
    st.subheader("📈 Scatter Plot Interaktif")
    x_axis = st.selectbox("Pilih sumbu X", options=data.columns[1:], index=0)
    y_axis = st.selectbox("Pilih sumbu Y", options=data.columns[1:], index=1)
    scatter_chart = alt.Chart(data).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color="Species",
        tooltip=["Species", x_axis, y_axis]
    ).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)
    
    # Add histogram
    st.subheader("📊 Histogram")
    feature = st.selectbox("Pilih Fitur untuk Histogram", options=data.columns[1:], index=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data[feature], bins=20, color="skyblue", edgecolor="black")
    ax.set_title(f"Histogram untuk {feature}")
    st.pyplot(fig)

    # Heatmap
    # Heatmap
st.subheader("🔥 Korelasi Fitur")
if st.checkbox("Tampilkan Heatmap Korelasi"):
    # Pilih hanya kolom numerik untuk menghitung korelasi
    numeric_columns = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)


# Model prediction
elif menu == "🤖 Model Prediksi":
    st.markdown("<h1 style='color: purple;'>Model Prediksi Berat Ikan</h1>", unsafe_allow_html=True)
    
    # Data preparation
    X = data.drop(columns=["Weight", "Species"])
    y = data["Weight"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)

    # Display results
    st.metric(label="📉 Mean Squared Error (MSE)", value=f"{mse:.2f}")
    
    # Prediction vs Actual
    st.subheader("Prediksi vs Aktual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    ax.set_title("Prediksi vs Aktual")
    st.pyplot(fig)

    # Input user
    st.subheader("🔢 Input Data Manual")
    user_inputs = {col: st.number_input(f"Masukkan {col}", value=0.0) for col in X.columns}
    
    if st.button("🔮 Prediksi Berat Ikan"):
        user_data = np.array(list(user_inputs.values())).reshape(1, -1)
        user_data_poly = poly.transform(user_data)
        user_prediction = model.predict(user_data_poly)[0]
        st.success(f"Berat ikan diprediksi: **{user_prediction:.2f}** gram")
