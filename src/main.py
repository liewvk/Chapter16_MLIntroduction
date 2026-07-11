import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    data = {
        "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Score": [35, 45, 50, 55, 65, 70, 75, 85, 88, 95]
    }

    df = pd.DataFrame(data)

    print("Student Dataset")
    print("---------------")
    print(df)

    X = df[["StudyHours"]]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "ActualScore": y_test,
        "PredictedScore": predictions
    })

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    new_student = pd.DataFrame({
        "StudyHours": [6.5]
    })

    predicted_score = model.predict(new_student)

    print()
    print(f"Predicted score for 6.5 study hours: {predicted_score[0]:.2f}")


main()
