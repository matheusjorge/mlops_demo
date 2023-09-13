from sklearn.model_selection import train_test_split

def preprocessing(data):
    X = data.drop(columns=["target"]).copy()
    y = data["target"].copy()

    return train_test_split(X, y, test_size=0.1)