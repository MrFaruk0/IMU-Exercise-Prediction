import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

def train(df, use_subject_session=False):
    df = df.copy()


    print(f"Orijinal veri boyutu: {df.shape}")
    print(f"Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    

    max_samples = 800000
    if len(df) > max_samples:
        print(f"Veri çok büyük, {max_samples:,} örnek alınıyor...")
        df = df.sample(n=max_samples, random_state=42)
        print(f"Örneklenmiş veri boyutu: {df.shape}")
        print(f"Yeni bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    if "Workout" not in df.columns:
        raise ValueError(f"'Workout' sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")

    drop_cols = ["Workout"]
    if not use_subject_session:
        for c in ("Subject", "Session"):
            if c in df.columns:
                drop_cols.append(c)

    X = df.drop(columns=drop_cols).values
    y = df["Workout"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4, verbose=2)
    model.fit(X_train, y_train)

    # Classification Report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, scaler, le