from datasets import load_dataset

SETS = [
    "deepset/prompt-injections",
    "Lakera/gandalf_ignore_instructions",
    "jackhhao/jailbreak-classification",
    "reshabhs/SPML_Chatbot_Prompt_Injection",
    "databricks/databricks-dolly-15k",
    "tatsu-lab/alpaca",
]

for name in SETS:
    try:
        ds = load_dataset(name)
        folder = "data/raw/" + name.replace("/", "__")
        ds.save_to_disk(folder)
        print("OK  ", name)
    except Exception as e:
        print("FAIL", name, type(e).__name__)
