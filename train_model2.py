import pandas as pd
import numpy as np
import esm
import torch
import joblib
import json
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

REFERENCE_SEQ = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIH"
    "VSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDP"
    "FLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPI"
    "NLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNE"
    "NGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYA"
    "WNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYK"
    "LPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYG"
    "FQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRD"
    "IADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGS"
    "NVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAI"
    "PTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQ"
    "IYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTV"
    "LPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKI"
    "QDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYV"
    "TQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAIC"
    "HDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYF"
    "KNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMV"
    "TIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
).replace("\n", "")

# Verify no non-ASCII characters in sequence
assert all(ord(c) < 128 for c in REFERENCE_SEQ), "Non-ASCII character in REFERENCE_SEQ!"
print(f"Reference sequence length: {len(REFERENCE_SEQ)} aa (verified pure ASCII)")


def apply_mutation(sequence, mutation):
    try:
        wt  = mutation[0]
        pos = int(mutation[1:-1]) - 1
        new = mutation[-1]
        if pos < 0 or pos >= len(sequence):
            return None
        if sequence[pos] != wt:
            return None
        return sequence[:pos] + new + sequence[pos + 1:]
    except Exception:
        return None


def get_embedding(sequence, model, batch_converter):
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        results = model(tokens, repr_layers=[6])
    return results["representations"][6].mean(1).squeeze().numpy()


print("Loading ESM-2 model ...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
print("  ESM-2 loaded.\n")

DATA_FILE = "train_data.csv"
if not os.path.exists(DATA_FILE):
    print(f"[ERROR] {DATA_FILE} not found. Run: python fetch_dms_data.py")
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} mutations from {DATA_FILE}\n")

print("Computing ESM-2 embeddings ... (this takes a few minutes)\n")
X_list, y_list, used_muts = [], [], []
skipped = 0

for i, row in df.iterrows():
    seq = apply_mutation(REFERENCE_SEQ, row["mutation"])
    if seq is None:
        skipped += 1
        continue
    emb = get_embedding(seq, esm_model, batch_converter)
    X_list.append(emb)
    y_list.append(float(row["escape_score"]))
    used_muts.append(row["mutation"])
    if (i + 1) % 50 == 0:
        print(f"  Embedded {i+1}/{len(df)} ...")

X = np.array(X_list)
y = np.array(y_list)
print(f"\n  Embedded : {len(X)} mutations  (skipped {skipped})")
print(f"  X shape  : {X.shape}")
print(f"  y range  : {y.min():.4f} - {y.max():.4f}\n")

print("Running 5-fold cross-validation ...")
ridge = Ridge(alpha=1.0)
kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(ridge, X, y, cv=kf, scoring="r2")
cv_mae = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_absolute_error")
print(f"  CV R2  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  CV MAE : {(-cv_mae).mean():.4f} +/- {(-cv_mae).std():.4f}\n")

print("Fitting final model on all data ...")
ridge.fit(X, y)
y_pred     = ridge.predict(X)
train_r2   = r2_score(y, y_pred)
train_mae  = mean_absolute_error(y, y_pred)
pearson_r, pearson_p = pearsonr(y, y_pred)
print(f"  Train R2    : {train_r2:.4f}")
print(f"  Train MAE   : {train_mae:.4f}")
print(f"  Pearson r   : {pearson_r:.4f}  (p={pearson_p:.2e})\n")

joblib.dump(ridge, "model.pkl")
print("  Saved model -> model.pkl")

metrics = {
    "n_mutations":  int(len(X)),
    "cv_r2_mean":   float(cv_r2.mean()),
    "cv_r2_std":    float(cv_r2.std()),
    "cv_mae_mean":  float((-cv_mae).mean()),
    "cv_mae_std":   float((-cv_mae).std()),
    "train_r2":     float(train_r2),
    "train_mae":    float(train_mae),
    "pearson_r":    float(pearson_r),
    "pearson_p":    float(pearson_p),
    "data_source":  "Curated from Greaney, Starr, Bloom, Cao et al. 2021-2022",
    "model":        "Ridge(alpha=1.0) on ESM-2 esm2_t6_8M_UR50D mean embeddings",
    "cv_folds":     5,
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Saved metrics -> model_metrics.json\n")

print("=" * 55)
print("  Training complete.")
print(f"  Training set   : {len(X)} mutations")
print(f"  CV R2 (5-fold) : {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")
print(f"  Pearson r      : {pearson_r:.3f}")
print(f"\n  Next: python -m streamlit run app.py")
print("=" * 55)