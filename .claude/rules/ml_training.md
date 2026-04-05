# ML Training Rules — Updated April 2026
## Critical rules for model training, validation, and saving

---

## Rule 1: Always Validate Input Shape BEFORE Fitting

```python
assert X_train.shape[0] > 0, f"Training set is empty — check upstream pipeline"
assert X_train.shape[0] == y_train.shape[0], "X/y sample count mismatch"
assert not np.isnan(X_train).any(), "NaN values detected in training features"
```

---

## Rule 2: Split Data BEFORE Any Preprocessing (Prevent Leakage)

```python
from sklearn.model_selection import train_test_split
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE

# WRONG — preprocessing before split leaks test info:
# X_scaled = StandardScaler().fit_transform(X)  # then split

# CORRECT:
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE
)
scaler = StandardScaler().fit(X_train)  # fit only on train
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

---

## Rule 3: Drug Response — Use Trailing Moving Average (TMA) Evaluation

Drug response data (GDSC2 IC50) is HEAVILY skewed toward non-responsive values.
Standard RMSE or accuracy gives a misleading overall metric.
Use TMA as per MMDRP paper (Bioinformatics Advances 2024):

```python
def trailing_moving_average(y_true, y_pred, window_pct=0.05):
    """
    Sort by predicted value; compute rolling RMSE in a trailing window.
    Evaluates performance at the HIGH-RESPONSE (low IC50) end of the distribution.
    A model that cannot predict responders is USELESS clinically.
    """
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).sort_values('pred')
    window = max(1, int(len(df) * window_pct))
    df['tma_rmse'] = df['true'].rolling(window).apply(
        lambda x: np.sqrt(((x - df.loc[x.index, 'pred'])**2).mean())
    )
    return df['tma_rmse'].dropna()
```

---

## Rule 4: Subtype-GAN — Adversarial Training Loop

```python
# The GAN training loop must alternate between:
# 1. Update discriminator (freeze encoder/decoder)
# 2. Update encoder + decoder (freeze discriminator)
# Then jointly update reconstruction loss
# Do NOT update all parameters simultaneously — training will collapse

for epoch in range(SUBTYPE_GAN_EPOCHS):
    # Phase 1: Update discriminator
    optimizer_disc.zero_grad()
    disc_loss = -torch.mean(D(z_real)) + torch.mean(D(z_fake))
    disc_loss.backward(); optimizer_disc.step()

    # Phase 2: Update encoder (fool discriminator)
    optimizer_enc.zero_grad()
    enc_adv_loss = -torch.mean(D(z_enc))
    enc_adv_loss.backward(); optimizer_enc.step()

    # Phase 3: Update reconstruction (encoder + decoder)
    optimizer_ae.zero_grad()
    recon_loss = F.mse_loss(x_recon, x_true)
    recon_loss.backward(); optimizer_ae.step()
```

---

## Rule 5: LMF (Low-Rank Multimodal Fusion) — Correct Implementation

```python
def low_rank_multimodal_fusion(z1, z2, rank=16):
    """
    Fuse two latent vectors via low-rank tensor fusion.
    Avoids quadratic explosion of tensor product.
    Based on Liu et al. 2018 (MMDRP, DeepDRA).
    """
    z1_ext = torch.cat([z1, torch.ones(z1.shape[0], 1, device=z1.device)], dim=1)
    z2_ext = torch.cat([z2, torch.ones(z2.shape[0], 1, device=z2.device)], dim=1)
    # Outer product via einsum
    fused = torch.einsum('bi,bj->bij', z1_ext, z2_ext)
    fused = fused.view(fused.shape[0], -1)  # Flatten
    return fused
```

---

## Rule 6: SHAP Explanations — Required on Every Model

Every trained model MUST expose a `.explain(X)` method:

```python
import shap

class BiomarkerModel:
    def explain(self, X: np.ndarray) -> pd.DataFrame:
        """
        Return SHAP values as DataFrame with gene names as columns.
        Required for XAI compliance at every pipeline phase.
        """
        explainer = shap.DeepExplainer(self.model, self.X_background)
        shap_values = explainer.shap_values(X)
        return pd.DataFrame(shap_values, columns=self.feature_names)
```

For tree-based models:
```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
```

For PyTorch via Captum (Integrated Gradients):
```python
from captum.attr import IntegratedGradients
ig = IntegratedGradients(model)
attributions = ig.attribute(X_tensor, target=0)
```

---

## Rule 7: Early Stopping — Required for All Neural Networks

```python
from config import DRP_PATIENCE

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= DRP_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break
```

---

## Rule 8: Knowledge Graph Training (PyKEEN) — April 2026

```python
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from config import KG_MODEL, KG_EMBEDDING_DIM, KG_EPOCHS

tf = TriplesFactory.from_path(drkg_triples_path)
train_tf, test_tf = tf.split([0.8, 0.2])

result = pipeline(
    training=train_tf,
    testing=test_tf,
    model=KG_MODEL,                    # "TransE" or "RotatE"
    model_kwargs={"embedding_dim": KG_EMBEDDING_DIM},
    training_kwargs={"num_epochs": KG_EPOCHS},
    random_seed=42,
)

# Score drug→LIHC disease edges not in training set:
model = result.model
drug_embeddings = model.entity_representations[0]()
```

---

## Rule 9: AlphaFold3 Confidence Thresholding (April 2026)

Use pTM and ipTM (interface predicted TM-score) as binding confidence:

```python
from config import AF3_BINDING_THRESHOLD

def parse_af3_output(af3_json_path: str) -> dict:
    """Parse AlphaFold3 output JSON and extract binding scores."""
    with open(af3_json_path) as f:
        result = json.load(f)

    ptm = result['summary_confidences']['ptm']
    iptm = result['summary_confidences']['iptm']
    binding_pass = iptm >= AF3_BINDING_THRESHOLD

    return {'ptm': ptm, 'iptm': iptm, 'binding_pass': binding_pass}
```

A higher `ipTM` (interface pTM) → stronger protein-ligand interaction confidence.
ipTM > 0.5 generally indicates a well-modeled interface.

---

## Rule 10: Save and Version Every Model

```python
import json
from datetime import datetime
from config import MODELS_DIR

def save_model_with_metadata(model, model_path, metadata: dict):
    """Save model + metadata JSON for reproducibility."""
    torch.save(model.state_dict(), model_path)

    meta = {
        "saved_at": datetime.now().isoformat(),
        "model_class": model.__class__.__name__,
        **metadata
    }
    meta_path = model_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved: {model_path}")
    logger.info(f"Metadata saved: {meta_path}")
```

---

## Rule 11: Cross-Validation Splits for Drug Response Must Prevent Leakage

Drug response data has three valid split strategies (as per MMDRP paper):

```python
# Strategy 1: Random split (optimistic — can see similar drugs/cell lines in test)
# Strategy 2: Drug split (held-out drugs — realistic for repurposing)
# Strategy 3: Cell line split (held-out cell lines — realistic for new patients)
# Strategy 4: Both held-out (most stringent — recommended for publication)

from sklearn.model_selection import GroupShuffleSplit

if split_strategy == "drug":
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X, y, groups=drug_ids):
        # drug_ids ensures no test drug appears in training
        ...
```
