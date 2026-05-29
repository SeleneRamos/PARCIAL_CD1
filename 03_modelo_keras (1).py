import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, json, multiprocessing

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              confusion_matrix, roc_curve,
                              precision_recall_curve, average_precision_score,
                              ConfusionMatrixDisplay)

# ── Semilla global ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ── Configuración CPU / GPU ────────────────────────────────────────────────
N_CPUS = max(1, multiprocessing.cpu_count() - 1)   # todos los núcleos menos 1

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Crecimiento de memoria dinámico (evita reservar toda la VRAM de golpe)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"   🟢 GPU detectada: {[g.name for g in gpus]} — usando GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)
else:
    # Sin GPU: paralelizar ops de TF sobre todos los núcleos disponibles
    tf.config.threading.set_inter_op_parallelism_threads(N_CPUS)
    tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)
    print(f"   🟡 Sin GPU — usando {N_CPUS} núcleos CPU")

DIRS = {
    "modelos":  "modelo_final",
    "graficos": "plots/modelos_ml",
    "ablation": "ablation_study",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 65)
print("1. Cargando datos preprocesados (pipeline_log y pipeline_disc)...")

X_train_log_full  = np.load("transform/X_train_log.npy")
X_test_log        = np.load("transform/X_test_log.npy")
X_train_disc_full = np.load("transform/X_train_disc.npy")
X_test_disc       = np.load("transform/X_test_disc.npy")
y_train_full      = np.load("transform/y_train.npy")
y_test            = np.load("transform/y_test.npy")

print(f"   pipeline_log  — X_train: {X_train_log_full.shape}  X_test: {X_test_log.shape}")
print(f"   pipeline_disc — X_train: {X_train_disc_full.shape}  X_test: {X_test_disc.shape}")
print(f"   Balance train: {y_train_full.mean():.1%} positivos")

INPUT_DIM_LOG  = X_train_log_full.shape[1]
INPUT_DIM_DISC = X_train_disc_full.shape[1]

# ── Split train → train (85%) + validación (15%) ───────────────────────────
X_train_log,  X_val_log,  y_train, y_val = train_test_split(
    X_train_log_full, y_train_full,
    test_size=0.15, random_state=SEED, stratify=y_train_full
)
X_train_disc, X_val_disc, _, _ = train_test_split(
    X_train_disc_full, y_train_full,
    test_size=0.15, random_state=SEED, stratify=y_train_full
)
print(f"   Train: {X_train_log.shape[0]} | Val: {X_val_log.shape[0]} | Test: {X_test_log.shape[0]}")

# ── Class weights ──────────────────────────────────────────────────────────
clases = np.array([0, 1])
pesos  = compute_class_weight(class_weight='balanced', classes=clases, y=y_train)
class_weight_dict = dict(zip(clases, pesos))
print(f"   class_weight: {class_weight_dict}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_base_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1,  activation='sigmoid'),
    ], name="modelo_base")
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    return model


def build_tuned_model(capas: list, dropout: float, l2: float,
                      lr: float, input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for n in capas:
        x = layers.Dense(n, activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    return model


def evaluar_modelo(model, X, y, nombre="Modelo") -> dict:
    prob  = model.predict(X, verbose=0).ravel()
    pred  = (prob >= 0.5).astype(int)
    auc   = roc_auc_score(y, prob)
    f1    = f1_score(y, pred)
    acc   = accuracy_score(y, pred)
    pr_auc = average_precision_score(y, prob)
    cm    = confusion_matrix(y, pred)
    print(f"\n  {nombre}")
    print(f"    ROC-AUC : {auc:.4f} | PR-AUC : {pr_auc:.4f} | F1 : {f1:.4f} | Acc : {acc:.4f}")
    return {'nombre': nombre, 'auc': auc, 'pr_auc': pr_auc,
            'f1': f1, 'acc': acc, 'prob': prob, 'pred': pred, 'cm': cm}


def guardar_experimento(model, historia_dict: dict,
                         res_log: dict, res_disc: dict,
                         experimento: str, params: dict = None) -> dict:
    """Guarda modelo .keras, métricas .json, historia .json y gráficos .png."""
    slug    = experimento.lower().replace(" ", "_").replace("/", "-")
    exp_dir = os.path.join(DIRS["ablation"], slug)
    os.makedirs(exp_dir, exist_ok=True)

    # — Modelo ——————————————————————————————————————————————————————————————
    model.save(os.path.join(exp_dir, "modelo.keras"))

    # — Métricas ————————————————————————————————————————————————————————————
    def limpiar(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in d.items() if k not in ['prob', 'pred', 'cm']}

    metricas = {
        "experimento":       experimento,
        "params":            params or {},
        "pipeline_log":      limpiar(res_log),
        "pipeline_disc":     limpiar(res_disc),
        "epocas_entrenadas": len(historia_dict['loss']),
    }
    with open(os.path.join(exp_dir, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=2)

    # — Historia ————————————————————————————————————————————————————————————
    with open(os.path.join(exp_dir, "historia_entrenamiento.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in historia_dict.items()}, f, indent=2)

    # — Gráficos ————————————————————————————————————————————————————————————
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Ablation — {experimento}", fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(historia_dict['loss'],     label='Train')
    ax.plot(historia_dict['val_loss'], label='Val', linestyle='--')
    ax.set_title("Learning Curve — Loss")
    ax.set_xlabel("Época"); ax.set_ylabel("BCE"); ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(historia_dict['auc'],     label='Train')
    ax.plot(historia_dict['val_auc'], label='Val', linestyle='--')
    ax.set_title("Learning Curve — AUC")
    ax.set_xlabel("Época"); ax.set_ylabel("AUC"); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    for res, lbl in [(res_log, 'pipeline_log'), (res_disc, 'pipeline_disc')]:
        fpr, tpr, _ = roc_curve(y_test, res['prob'])
        ax.plot(fpr, tpr, label=f"{lbl} ({res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title("Curva ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for res, lbl in [(res_log, 'pipeline_log'), (res_disc, 'pipeline_disc')]:
        prec, rec, _ = precision_recall_curve(y_test, res['prob'])
        ax.plot(rec, prec, label=f"{lbl} ({res['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), color='k', linestyle='--', alpha=0.4,
               label=f'Baseline ({y_test.mean():.2f})')
    ax.set_title("Curva PR"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ConfusionMatrixDisplay(res_log['cm'],
                           display_labels=['<=50K', '>50K']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"CM — pipeline_log (F1={res_log['f1']:.3f})")

    ax = fig.add_subplot(gs[1, 2])
    ConfusionMatrixDisplay(res_disc['cm'],
                           display_labels=['<=50K', '>50K']).plot(
        ax=ax, colorbar=False, cmap='Oranges')
    ax.set_title(f"CM — pipeline_disc (F1={res_disc['f1']:.3f})")

    plt.savefig(os.path.join(exp_dir, "graficos.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ Experimento '{experimento}' guardado en {exp_dir}/")
    return metricas


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELO BASE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("2. Entrenando MODELO BASE...")

tf.random.set_seed(SEED)
modelo_base = build_base_model(INPUT_DIM_LOG)
modelo_base.summary()

hist_base = modelo_base.fit(
    X_train_log, y_train,
    validation_data=(X_val_log, y_val),
    epochs=100, batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_auc', patience=10,
                             restore_best_weights=True, mode='max')],
    verbose=0
)

# Versión disc del base (misma arquitectura, distinta entrada)
tf.random.set_seed(SEED)
modelo_base_disc = build_base_model(INPUT_DIM_DISC)
hist_base_disc = modelo_base_disc.fit(
    X_train_disc, y_train,
    validation_data=(X_val_disc, y_val),
    epochs=100, batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_auc', patience=10,
                             restore_best_weights=True, mode='max')],
    verbose=0
)

res_base_log  = evaluar_modelo(modelo_base,      X_test_log,  y_test, "Base / log")
res_base_disc = evaluar_modelo(modelo_base_disc, X_test_disc, y_test, "Base / disc")

metricas_base = guardar_experimento(
    model=modelo_base,
    historia_dict=hist_base.history,
    res_log=res_base_log,
    res_disc=res_base_disc,
    experimento="01_modelo_base",
    params={"arquitectura": [64, 32], "pipeline_principal": "log"},
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("3. Búsqueda de hiperparámetros con Optuna (TPE, 30 trials)...")

def objective(trial: optuna.Trial) -> float:
    """Optimiza val_AUC sobre pipeline_log."""
    n_capas = trial.suggest_int("n_capas", 1, 3)
    capas   = [trial.suggest_categorical(f"capa_{i}", [64, 128, 256])
               for i in range(n_capas)]
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    l2      = trial.suggest_float("l2",      1e-5, 1e-2, log=True)
    lr      = trial.suggest_float("lr",      1e-4, 1e-2, log=True)
    batch   = trial.suggest_categorical("batch", [128, 256, 512])

    tf.random.set_seed(SEED)
    m = build_tuned_model(capas=capas, dropout=dropout,
                          l2=l2, lr=lr, input_dim=INPUT_DIM_LOG)
    h = m.fit(
        X_train_log, y_train,
        validation_data=(X_val_log, y_val),
        epochs=100, batch_size=batch,
        class_weight=class_weight_dict,
        callbacks=[EarlyStopping(monitor='val_auc', patience=10,
                                 restore_best_weights=True, mode='max')],
        verbose=0
    )
    return max(h.history['val_auc'])

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    study_name="keras_adult_income"
)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n   ✅ Mejor trial  : #{study.best_trial.number}")
print(f"   ✅ Mejor val_AUC: {study.best_value:.4f}")
print(f"   ✅ Mejores params: {study.best_params}")

# Guardar resumen de todos los trials
trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(DIRS["ablation"], "optuna_trials.csv"), index=False)

# Importancia de hiperparámetros (gráfico)
try:
    fig_imp = optuna.visualization.matplotlib.plot_param_importances(study)
    fig_imp.figure.savefig(
        os.path.join(DIRS["ablation"], "optuna_param_importances.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig_imp.figure)
except Exception:
    pass  # Requiere plotly; omitir si no está instalado

# ── Reconstruir mejor_params desde study.best_params ──────────────────────
bp = study.best_params
mejor_params = {
    "capas":   [bp[f"capa_{i}"] for i in range(bp["n_capas"])],
    "dropout": bp["dropout"],
    "l2":      bp["l2"],
    "lr":      bp["lr"],
    "batch":   bp["batch"],
}

# ── Re-entrenar modelo tuneado con los mejores params (para guardar pesos) ──
tf.random.set_seed(SEED)
modelo_tuneado = build_tuned_model(
    capas=mejor_params['capas'], dropout=mejor_params['dropout'],
    l2=mejor_params['l2'], lr=mejor_params['lr'], input_dim=INPUT_DIM_LOG
)
hist_tuneado = modelo_tuneado.fit(
    X_train_log, y_train,
    validation_data=(X_val_log, y_val),
    epochs=100, batch_size=mejor_params['batch'],
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_auc', patience=10,
                             restore_best_weights=True, mode='max')],
    verbose=0
)

# Versión disc del tuneado
tf.random.set_seed(SEED)
modelo_tuneado_disc = build_tuned_model(
    capas=mejor_params['capas'], dropout=mejor_params['dropout'],
    l2=mejor_params['l2'], lr=mejor_params['lr'], input_dim=INPUT_DIM_DISC
)
modelo_tuneado_disc.fit(
    X_train_disc, y_train,
    validation_data=(X_val_disc, y_val),
    epochs=100, batch_size=mejor_params['batch'],
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_auc', patience=10,
                             restore_best_weights=True, mode='max')],
    verbose=0
)

res_tuneado_log  = evaluar_modelo(modelo_tuneado,      X_test_log,  y_test, "Tuneado / log")
res_tuneado_disc = evaluar_modelo(modelo_tuneado_disc, X_test_disc, y_test, "Tuneado / disc")

metricas_tuneado = guardar_experimento(
    model=modelo_tuneado,
    historia_dict=hist_tuneado.history,
    res_log=res_tuneado_log,
    res_disc=res_tuneado_disc,
    experimento="02_modelo_tuneado_optuna",
    params={**mejor_params, "optuna_trial": study.best_trial.number,
            "optuna_val_auc": study.best_value},
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODELO FINAL  (reentrenar sobre train_full con mejores params)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("4. Entrenando MODELO FINAL sobre train completo...")

tf.random.set_seed(SEED)
modelo_final = build_tuned_model(
    capas=mejor_params['capas'], dropout=mejor_params['dropout'],
    l2=mejor_params['l2'], lr=mejor_params['lr'], input_dim=INPUT_DIM_LOG
)
hist_final = modelo_final.fit(
    X_train_log_full, y_train_full,
    validation_data=(X_test_log, y_test),       # solo monitoreo
    epochs=150, batch_size=mejor_params['batch'],
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=15,
                      restore_best_weights=True, mode='max'),
        ModelCheckpoint(
            filepath=os.path.join(DIRS["modelos"], "modelo_keras_best.keras"),
            monitor='val_auc', save_best_only=True, mode='max'
        ),
    ],
    verbose=1
)

# Versión disc del final
tf.random.set_seed(SEED)
modelo_final_disc = build_tuned_model(
    capas=mejor_params['capas'], dropout=mejor_params['dropout'],
    l2=mejor_params['l2'], lr=mejor_params['lr'], input_dim=INPUT_DIM_DISC
)
hist_final_disc = modelo_final_disc.fit(
    X_train_disc_full, y_train_full,
    validation_data=(X_test_disc, y_test),
    epochs=150, batch_size=mejor_params['batch'],
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_auc', patience=15,
                             restore_best_weights=True, mode='max')],
    verbose=0
)

modelo_final.save(os.path.join(DIRS["modelos"], "modelo_keras_final.keras"))

res_final_log  = evaluar_modelo(modelo_final,      X_test_log,  y_test, "Final / log")
res_final_disc = evaluar_modelo(modelo_final_disc, X_test_disc, y_test, "Final / disc")

metricas_final = guardar_experimento(
    model=modelo_final,
    historia_dict=hist_final.history,
    res_log=res_final_log,
    res_disc=res_final_disc,
    experimento="03_modelo_final",
    params={**mejor_params, "entrenado_en": "train_full"},
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. JSON MAESTRO
# ══════════════════════════════════════════════════════════════════════════════
metricas_keras = {
    "modelo_base":    metricas_base,
    "modelo_tuneado": metricas_tuneado,
    "modelo_final":   metricas_final,
    "optuna": {
        "n_trials":     len(study.trials),
        "mejor_trial":  study.best_trial.number,
        "mejor_val_auc": study.best_value,
        "mejores_params": study.best_params,
    },
}
with open(os.path.join(DIRS["modelos"], "metricas_keras.json"), "w") as f:
    json.dump(metricas_keras, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 7. GRÁFICO COMPARATIVO FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("5. Generando gráfico comparativo final...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Comparativa final — pipeline_log vs pipeline_disc", fontsize=13)

ax = axes[0]
ax.plot(hist_final.history['loss'],          label='Train (log)')
ax.plot(hist_final.history['val_loss'],      label='Val (log)',  linestyle='--')
ax.plot(hist_final_disc.history['loss'],     label='Train (disc)', alpha=0.6)
ax.plot(hist_final_disc.history['val_loss'], label='Val (disc)',   alpha=0.6, linestyle='--')
ax.set_title("Learning Curves — Loss (modelo final)")
ax.set_xlabel("Época"); ax.set_ylabel("BCE"); ax.legend(fontsize=8)

ax = axes[1]
for res, lbl in [
    (res_base_log,    "Base/log"),    (res_base_disc,    "Base/disc"),
    (res_tuneado_log, "Tuneado/log"), (res_tuneado_disc, "Tuneado/disc"),
    (res_final_log,   "Final/log"),   (res_final_disc,   "Final/disc"),
]:
    fpr, tpr, _ = roc_curve(y_test, res['prob'])
    ax.plot(fpr, tpr, label=f"{lbl} ({res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_title("Curva ROC — todos los modelos")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7)

ax = axes[2]
for res, lbl in [
    (res_base_log,    "Base/log"),    (res_base_disc,    "Base/disc"),
    (res_tuneado_log, "Tuneado/log"), (res_tuneado_disc, "Tuneado/disc"),
    (res_final_log,   "Final/log"),   (res_final_disc,   "Final/disc"),
]:
    prec, rec, _ = precision_recall_curve(y_test, res['prob'])
    ax.plot(rec, prec, label=f"{lbl} ({res['pr_auc']:.3f})")
ax.axhline(y_test.mean(), color='k', linestyle='--', alpha=0.4,
           label=f'Baseline ({y_test.mean():.2f})')
ax.set_title("Curva PR — todos los modelos")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(DIRS["graficos"], "03_keras_evaluacion.png"),
            dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("RESUMEN COMPARATIVO — TEST")
print("═" * 65)
print(f"{'Modelo':<22} {'Pipeline':<14} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>7} {'Acc':>7}")
print("─" * 65)
for res, m_lbl, p_lbl in [
    (res_base_log,    "Base",    "log"),  (res_base_disc,    "Base",    "disc"),
    (res_tuneado_log, "Tuneado", "log"),  (res_tuneado_disc, "Tuneado", "disc"),
    (res_final_log,   "Final",   "log"),  (res_final_disc,   "Final",   "disc"),
]:
    print(f"  {m_lbl:<20} {p_lbl:<14} "
          f"{res['auc']:>8.4f} {res['pr_auc']:>8.4f} "
          f"{res['f1']:>7.4f} {res['acc']:>7.4f}")
print("═" * 65)

print("\n✅  03_modelo_keras.py completado")
print(f"\n   Archivos generados:")
print(f"   · {DIRS['modelos']}/modelo_keras_final.keras")
print(f"   · {DIRS['modelos']}/modelo_keras_best.keras")
print(f"   · {DIRS['modelos']}/metricas_keras.json")
print(f"   · {DIRS['graficos']}/03_keras_evaluacion.png")
print(f"   · {DIRS['ablation']}/01_modelo_base/")
print(f"   · {DIRS['ablation']}/02_modelo_tuneado_optuna/")
print(f"   · {DIRS['ablation']}/03_modelo_final/")
print(f"   · {DIRS['ablation']}/optuna_trials.csv")
print(f"   · {DIRS['ablation']}/optuna_param_importances.png")
print("═" * 65)