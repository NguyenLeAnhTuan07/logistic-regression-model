def encode_categorical(df, training= True):
    if training and "class" in df.columns:
        df["class"] = df["class"].map({
                "Normal": 0,
                "Abnormal": 1
            })
    return df
