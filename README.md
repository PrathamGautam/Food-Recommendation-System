# 🍽️ Food Recommendation System

Welcome to the **Food Recommendation System**! This project provides personalized meal suggestions based on user inputs like age, height, weight, and dietary preferences (Veg/Non-Veg). The system uses machine learning algorithms to offer recommendations tailored for breakfast, lunch, and dinner. Perfect for individuals seeking healthier food choices based on their Body Mass Index (BMI).

## 🚀 Features

- 🌱 **Personalized Meal Suggestions**: Get meal recommendations for breakfast, lunch, and dinner based on your unique inputs.
- 🧠 **Machine Learning**: Leverages K-Means clustering and Random Forest classifier to provide accurate food recommendations.
- 🧮 **BMI Calculation**: Calculates and considers your BMI to provide the best suggestions.
- 💻 **Streamlit UI**: User-friendly interface for easy input and interaction.

## 🛠️ Technologies Used

- **Python** 🐍
- **Pandas** 📊
- **NumPy** ➗
- **Scikit-learn** 🧠
- **Streamlit** 🌐
- **CSV Data** 📄

## ⚙️ Installation

Follow these steps to get the application up and running:

1. **Clone the repository** 📂:
    ```bash
    git clone https://github.com/your-username/food-recommendation-system.git
    cd food-recommendation-system
    ```

2. **Install the dependencies** 📦:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application** 🚀:
    ```bash
    streamlit run app.py
    ```

## 💡 How It Works

1. Enter your **age**, **weight**, **height**, and **dietary preferences** (Veg/Non-Veg).
2. The system calculates your **BMI** and suggests the most suitable food options for breakfast, lunch, and dinner.
3. Recommendations are generated using **K-Means clustering** and **Random Forest** based on a preloaded dataset of food items.
4. View your recommendations in a clean, user-friendly interface built with **Streamlit**.

## 📂 Project Structure

```bash
📦 food-recommendation-system
 ┣ 📂 data
 ┃ ┣ 📜 food.csv                   # Dataset containing food items and nutritional values
 ┣ 📜 app.py                        # Main Streamlit app
 ┣ 📜 requirements.txt              # Python package dependencies
 ┣ 📜 README.md                     # Project documentation
```
## 🌟 Future Scope
- 📱 Mobile App: Develop a mobile application for easier access on the go.
- ⌚ Wearable Integration: Sync with health trackers for more accurate recommendations.
- 🍲 Extended Food Database: Expand to include more cuisines and dietary options.
- 🧠 Advanced Models: Use deep learning techniques for even better food predictions.

## 📌 Conclusion
The Food Recommendation System leverages machine learning and user-specific data to provide personalized meal suggestions, promoting healthier food choices tailored to individual needs. By considering factors such as age, weight, height, and dietary preferences, this system aims to support users in achieving their health and fitness goals. With a user-friendly interface powered by Streamlit, it offers a seamless experience for users to receive customized food recommendations. As the project evolves, it holds the potential for expansion through additional features like mobile integration, larger food databases, and advanced recommendation algorithms, making it a valuable tool for diet management and healthy living.
