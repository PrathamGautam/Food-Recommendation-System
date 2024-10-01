import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('food.csv')
Breakfastdata = data['Breakfast']
BreakfastdataNumpy = Breakfastdata.to_numpy()

Lunchdata = data['Lunch']
LunchdataNumpy = Lunchdata.to_numpy()

Dinnerdata = data['Dinner']
DinnerdataNumpy = Dinnerdata.to_numpy()

Food_itemsdata = data['Food_items']

# Function to calculate BMI
def calculate_bmi(weight, height):
    bmi = weight / ((height / 100) ** 2)
    return bmi

# Function to get the food recommendations
def get_recommendations(age, veg, weight, height, meal):
    bmi = calculate_bmi(weight, height)

    if bmi < 16:
        clbmi = 4
    elif bmi >= 16 and bmi < 18.5:
        clbmi = 3
    elif bmi >= 18.5 and bmi < 25:
        clbmi = 2
    elif bmi >= 25 and bmi < 30:
        clbmi = 1
    elif bmi >= 30:
        clbmi = 0

    agecl = age // 20

    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    # retrieving rows by loc method
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[:, 5:15]

    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[:, 5:15]

    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[:, 5:15]

    # K-Means Based Dinner Food
    Datacalorie = DinnerfoodseparatedIDdata.to_numpy()

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dnrlbl = kmeans.labels_

    # K-Means Based lunch Food
    Datacalorie = LunchfoodseparatedIDdata.to_numpy()

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    lnchlbl = kmeans.labels_

    # K-Means Based breakfast Food
    Datacalorie = breakfastfoodseparatedIDdata.to_numpy()

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    brklbl = kmeans.labels_

    # Reading of the dataset
    datafin = pd.read_csv('nutrition_distriution.csv')

    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]].T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]].T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]].T

    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()

    weightlossfin = np.zeros((len(weightlosscatDdata) * 5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincatDdata) * 5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycatDdata) * 5, 9), dtype=np.float32)

    t, r, s = 0, 0, 0
    yt, yr, ys = [], [], []

    for zz in range(5):
        for jj in range(len(weightlosscatDdata)):
            valloc = list(weightlosscatDdata[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1

        for jj in range(len(weightgaincatDdata)):
            valloc = list(weightgaincatDdata[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1

        for jj in range(len(healthycatDdata)):
            valloc = list(healthycatDdata[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightlosscatDdata), 6), dtype=np.float32)

    for jj in range(len(weightlosscatDdata)):
        valloc = list(weightlosscatDdata[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)

    X_train = weightlossfin  # Features
    y_train = yt  # Labels

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    if meal == 'Breakfast':
        return [Food_itemsdata[i] for i in range(len(y_pred)) if y_pred[i] == 2]
    elif meal == 'Lunch':
        return [Food_itemsdata[i] for i in range(len(y_pred)) if y_pred[i] == 1]
    elif meal == 'Dinner':
        return [Food_itemsdata[i] for i in range(len(y_pred)) if y_pred[i] == 0]

# Streamlit app
st.title("Food Recommendation System")
st.image("food.webp")

st.sidebar.title("Enter your details")
age = st.sidebar.number_input("Age", min_value=1, max_value=100)
veg = st.sidebar.selectbox("Veg/Non-Veg", [1, 0])
weight = st.sidebar.number_input("Weight (in kg)", min_value=1, max_value=100)
height = st.sidebar.number_input("Height (in cm)", min_value=1, max_value=200)

meal = st.sidebar.selectbox("Meal", ["Breakfast", "Lunch", "Dinner"])

if st.sidebar.button("Get Recommendations"):
    recommendations = get_recommendations(age, veg, weight, height, meal)
    st.write("Recommended food items:")
    for item in recommendations:
        st.write(item)
