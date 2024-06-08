import pandas as pd
import sys

filename = f"data/{sys.argv[1]}"

# TODO: label the open hand samples
# step 0: get dataframe
data = pd.read_csv(f"{filename}.csv")
print(data)
# get size of df
print(data.shape)
numCol = data.shape[0]


newdata = [data.iloc[idx - 3 : idx + 3, :3].sum().sum() / 7 for idx in range(3, numCol)]
print(newdata)
maxD = max(newdata)
minD = min(newdata)
print(minD, maxD)
threshold = 0.63
clone = data.copy()
for idx in range(3, numCol):
    val = data.iloc[idx - 3 : idx + 3, :3].sum().sum() / 7
    # step 1: normalize (avg) 0 - 1
    normalized = (val - minD) / (maxD - minD)
    print(normalized)
    # step 2: compare
    if normalized > threshold:
        # step 3: threshold -> label
        clone.loc[idx, "label"] = "close"
    else:
        clone.loc[idx, "label"] = "open"
clone.to_csv(f"{filename}_processed.csv", index=False)

# open_hand = data[data["label"] == "open"]
# closed_hand = data[data["label"] == "close"]
