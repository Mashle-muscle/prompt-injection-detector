from datasets import load_from_disk

ds = load_from_disk("data/raw/deepset__prompt-injections")
print(ds)
print(ds["train"][0])
from datasets import load_from_disk

ds = load_from_disk("data/raw/deepset__prompt-injections")

# what the splits look like
print(ds)

# first row
print(ds["train"][0])

# first malicious row — confirms label polarity
for row in ds["train"]:
    if row["label"] == 1:
        print("MALICIOUS:", row["text"][:200])
        break
import os

for folder in os.listdir("data/raw"):
    path = os.path.join("data/raw", folder)
    if not os.path.isdir(path):
        continue
    try:
        d = load_from_disk(path)
        split = list(d.keys())[0]
        print(f"{folder:45} cols={d[split].column_names} rows={len(d[split])}")
    except Exception as e:
        print(f"{folder:45} SKIP {type(e).__name__}")
from datasets import load_from_disk

# 1. what values does jackhhao's 'type' column take?
d = load_from_disk("data/raw/jackhhao__jailbreak-classification")
print("jackhhao types:", set(r["type"] for r in d["train"]))

# 2. what does SPML's 'Prompt injection' column contain?
s = load_from_disk("data/raw/reshabhs__SPML_Chatbot_Prompt_Injection")
print("SPML labels:", set(r["Prompt injection"] for r in s["train"]))
print("SPML sample:", s["train"][0])
import pandas as pd
from datasets import load_from_disk

def rows_from(folder, text_fn, label_fn, source):
    ds = load_from_disk(f"data/raw/{folder}")
    out = []
    for split in ds:
        for r in ds[split]:
            t = text_fn(r)
            if t and len(t.strip()) > 3:
                out.append({"text": t.strip(), "label": label_fn(r), "source": source})
    return out

all_rows = []

# deepset — already clean
all_rows += rows_from("deepset__prompt-injections",
                      lambda r: r["text"], lambda r: int(r["label"]), "deepset")

# gandalf — every row is an attack
all_rows += rows_from("Lakera__gandalf_ignore_instructions",
                      lambda r: r["text"], lambda r: 1, "gandalf")

# dolly — all benign
all_rows += rows_from("databricks__databricks-dolly-15k",
                      lambda r: r["instruction"] + (" " + r["context"] if r["context"] else ""),
                      lambda r: 0, "dolly")

df = pd.DataFrame(all_rows).drop_duplicates(subset="text")
print(df.groupby(["source", "label"]).size())
from datasets import load_from_disk
d = load_from_disk("data/raw/jackhhao__jailbreak-classification")
print(set(r["type"] for r in d["train"]))
print(d["train"][0])
