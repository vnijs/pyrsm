import pandas as pd
import os


def convert(directory):
    # convert all pkl files in a directory to parquet and save the description
    pkf = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    for f in pkf:
        print("Converting " + f + " to parquet")
        df = pd.read_pickle(os.path.join(directory, f))
        with open(
            os.path.join(directory, f.replace(".pkl", "_description.md")), "w"
        ) as file:
            file.write(df.description)
        df.to_parquet(os.path.join(directory, f.replace(".pkl", ".parquet")))
        os.remove(os.path.join(directory, f.replace(".pkl", ".prq")))


convert("data")
convert("basics")
convert("design")
convert("model")
convert("multivariate")
