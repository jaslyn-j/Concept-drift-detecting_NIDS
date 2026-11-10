import json

v = json.load(open("data/meta.json"))["versions"]
print("pip install " + " ".join([
    f"scikit-learn=={v['sklearn']}",
    f"xgboost=={v['xgboost']}",
    f"pandas=={v['pandas']}",
    f"numpy=={v['numpy']}",
]))