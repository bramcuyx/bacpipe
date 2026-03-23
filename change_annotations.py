import pandas as pd


df_annot = pd.read_csv("light_20260319_dataset_denoised.csv")
df_singleannot = pd.DataFrame(columns=df_annot.columns)
print(df_annot.columns)
for audiofilename in df_annot.audiofilename.unique():
    sel = df_annot[df_annot.audiofilename==audiofilename]
    sel.loc[:, "end"] = 10.0
    sel.loc[:, "start"] = 0.0
    if len(sel) > 1:
        # for every audio file keep one row with the annotation 1 if any segment is annotated as 1, and 0 otherwise
        sel.loc[:, "label:event"] = int(sel['label:event'].sum() > 0)
        df_singleannot = pd.concat([df_singleannot, sel.iloc[0:1]])

    else:
        df_singleannot = pd.concat([df_singleannot, sel])
df_singleannot.to_csv("light_20260319_dataset_denoised_singleannot.csv", index=False)

