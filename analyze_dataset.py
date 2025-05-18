from pathlib import Path
import pandas as pd

base_dir = Path("archive/test")

class_counts = {}
for class_dir in base_dir.iterdir():
    if class_dir.is_dir():
        count = len(list(class_dir.glob("*.jpg")))
        class_counts[class_dir.name] = count

df_classes = pd.DataFrame.from_dict(class_counts, orient="index", columns=["Bildanzahl"]).sort_values(by="Bildanzahl", ascending=False)
df_classes.index.name = "Klasse"

print(df_classes)