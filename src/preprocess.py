import pandas as pd

def preprocess(path="data/RGDT/RecGym.csv"):

    df = pd.read_csv(path)

    #Position yok
    df = df[["Subject", "Session", "Workout", "A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1", "Workout"]]
    #Null çıkar
    df.dropna(inplace=True)

    return df

