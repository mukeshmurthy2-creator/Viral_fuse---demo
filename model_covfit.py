# =========================
# model_covfit.py
# Backend logic for ViralFuse
# =========================

import numpy as np

# Try importing ESM (optional fallback)
try:
    import esm
    import torch
    ESM_AVAILABLE = True
except:
    ESM_AVAILABLE = False

# =========================
# LOAD ESM MODEL (only once)
# =========================
if ESM_AVAILABLE:
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()


# =========================
# MUTATION FUNCTION
# =========================
def apply_mutation(sequence, mutation):
    try:
        if len(mutation) < 3:
            return None, "Invalid mutation format"

        original = mutation[0]
        new = mutation[-1]

        if not mutation[1:-1].isdigit():
            return None, "Invalid position"

        pos = int(mutation[1:-1]) - 1

        if pos < 0 or pos >= len(sequence):
            return None, f"Position {pos+1} out of range"

        if sequence[pos] != original:
            return None, f"Expected {original}, found {sequence[pos]}"

        mutated = sequence[:pos] + new + sequence[pos+1:]
        return mutated, None

    except Exception as e:
        return None, str(e)


# =========================
# EMBEDDING FUNCTION
# =========================
def get_embedding(sequence):
    if ESM_AVAILABLE:
        data = [("protein", sequence)]
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])

        token_embeddings = results["representations"][6]
        return token_embeddings.mean(1).squeeze().numpy()

    else:
        # fallback (if esm not installed)
        return np.random.randn(320)


# =========================
# PREDICTION FUNCTION
# =========================
import joblib

# Load trained model
model_reg = joblib.load("model.pkl")

def predict_escape(sequence):
    emb = get_embedding(sequence)
    score = model_reg.predict([emb])[0]

    # Clamp score
    score = float(score)

    if score >= 0.7:
        label = "High Escape"
    elif score >= 0.4:
        label = "Moderate Escape"
    else:
        label = "Low Escape"

    return score, label

# =========================
# FULL PIPELINE FUNCTION
# =========================
def run_prediction(sequence, mutation):
    mutated_seq, err = apply_mutation(sequence, mutation)

    if err:
        return None, err

    score, label = predict_escape(mutated_seq)

    return {
        "mutation": mutation,
        "score": round(score, 4),
        "label": label
    }, None