import pandas as pd
import os


def convert(base_dir):
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for f in filenames:
            if f.endswith(".pkl"):
                print(f"Converting {f} to parquet")
                df = pd.read_pickle(os.path.join(dirpath, f))
                df.to_parquet(os.path.join(dirpath, f.replace(".pkl", ".parquet")))
                if hasattr(df, "description"):
                    fmd = f.replace(".pkl", "_description.md")
                    md_file = os.path.join(dirpath, fmd)
                    with open(md_file, "w") as file:
                        print(f"Storing description in {fmd}")
                        file.write(df.description)


base_path = os.path.expanduser("~/gh/pyrsm/pyrsm/data")
if os.path.exists(base_path):
    convert(base_path)
