import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("hospital_appointments.csv", parse_dates=["date"])
    return df

df = load_data()

# ---------------------------
# Feature Engineering
# ---------------------------
df["day_of_week"] = df["date"].dt.day_name()

# ---------------------------
# Title
# ---------------------------
st.title("🏥 Advanced Healthcare Dashboard")

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("📂 Filters")

min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

department = st.sidebar.multiselect(
    "Department",
    df["department"].unique(),
    default=df["department"].unique()
)

filtered_df = df[
    (df["department"].isin(department)) &
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# ---------------------------
# 📈 Trend Line
# ---------------------------
st.subheader("📈 Appointments Over Time")

trend = filtered_df.groupby("date").size()

fig1, ax1 = plt.subplots()
trend.plot(ax=ax1)
ax1.set_ylabel("Appointments")
st.pyplot(fig1)

# ---------------------------
# 🔥 Heatmap (Day vs No-Show)
# ---------------------------
st.subheader("🔥 No-Show Heatmap (Day vs Count)")

heatmap_data = filtered_df[filtered_df["status"] == "No-Show"]
heatmap_data = heatmap_data.groupby("day_of_week").size()

# Convert to DataFrame
heatmap_df = heatmap_data.reindex([
    "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
]).fillna(0)

fig2, ax2 = plt.subplots()
sns.heatmap(
    heatmap_df.values.reshape(-1,1),
    annot=True,
    yticklabels=heatmap_df.index,
    xticklabels=["No-Shows"],
    cmap="Reds",
    ax=ax2
)
st.pyplot(fig2)

# ---------------------------
# 🤖 No-Show Prediction Model
# ---------------------------
st.subheader("🤖 Predict No-Show")

model_df = df.copy()

# Encode categorical data
le_gender = LabelEncoder()
le_dept = LabelEncoder()
le_status = LabelEncoder()

model_df["gender"] = le_gender.fit_transform(model_df["gender"])
model_df["department"] = le_dept.fit_transform(model_df["department"])
model_df["status"] = model_df["status"].apply(lambda x: 1 if x=="No-Show" else 0)

# Features & target
X = model_df[["patient_age", "gender", "department"]]
y = model_df["status"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# User Input for Prediction
# ---------------------------
st.write("### Enter Patient Details")

age = st.slider("Age", 0, 100, 30)
gender = st.selectbox("Gender", le_gender.classes_)
dept = st.selectbox("Department", le_dept.classes_)

# Encode input
input_data = pd.DataFrame({
    "patient_age": [age],
    "gender": [le_gender.transform([gender])[0]],
    "department": [le_dept.transform([dept])[0]]
})

# Predict
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

if st.button("Predict"):
    if prediction == 1:
        st.error(f"⚠️ Likely No-Show (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Likely to Attend (Probability: {1-prob:.2f})")