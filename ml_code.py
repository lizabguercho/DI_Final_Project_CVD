from sklearn.linear_model import LogisticRegression

result = train_binary_classifier(
    model=LogisticRegression(max_iter=10000000),
    df=df,
    target="cardio",          # change to your target column
    features=None,                 # or a list like ["age","bmi","bp", ...]
    test_size=0.3,
    random_state=42,
)

# Access metrics
result["metrics"]["auc"]

pipe = make_logreg_pipeline(df, target="cardio", use_balanced=False)
result = train_binary_classifier(
    model=pipe,
    df=df,
    target="cardio",
    test_size=0.2,
    random_state=42,
    stratify=True