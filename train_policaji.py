import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

TRAIN_CSV = "train_dataset.csv"
MODEL_IZHOD = "model_policaji.pkl"
GRAF_IZHOD = "grafi_policaji.png"

def treniraj_in_shrani_policaje():
    print("--- 1. UČENJE MODELA ZA LEŽEČE POLICAJE ---")
    
    try:
        df = pd.read_csv(TRAIN_CSV)
    except FileNotFoundError:
        print(f"NAPAKA: Datoteka {TRAIN_CSV} ne obstaja!")
        return
    
    ne_znacilke = ['target', 'cas_okna_start']
    feature_cols = [c for c in df.columns if c not in ne_znacilke]
    
    X = df[feature_cols]
    y = df['target']
    
    st_negativnih = len(y[y == 0])
    st_pozitivnih = len(y[y == 1])
    razmerje = st_negativnih / st_pozitivnih if st_pozitivnih > 0 else 1.0
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"   Model se uči na {len(feature_cols)} značilkah...")
    
    # POPRAVKI ZA OVERFITTING (max_depth=3, subsample, colsample_bytree, early_stopping)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=razmerje,
        use_label_encoder=False,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print("\n--- 2. OCENA MODELA ---")
    preds_val = model.predict(X_val)
    print(f"Najboljša iteracija: {model.best_iteration}")
    print(classification_report(y_val, preds_val))
    
    # Shranjevanje modela
    paket_za_shranjevanje = {
        'model': model,
        'features': feature_cols
    }
    joblib.dump(paket_za_shranjevanje, MODEL_IZHOD)
    print(f"Model uspešno shranjen v '{MODEL_IZHOD}'.")

    # --- 3. IZRIS GRAFOV ---
    print("\n--- 3. GENERIRANJE GRAFIKOV ---")
    rezultati = model.evals_result()
    epochs = len(rezultati['validation_0']['logloss'])
    x_os = range(0, epochs)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RoadQualityAi - Evaluacija modela za Ležeče policaje', fontsize=16, fontweight='bold', color='#00796b')

    # Graf 1: Log Loss
    ax[0].plot(x_os, rezultati['validation_0']['logloss'], label='Učna množica', color='#2196f3')
    ax[0].plot(x_os, rezultati['validation_1']['logloss'], label='Validacijska množica', color='#00796b')
    ax[0].set_title('Krivulja izgube (Log Loss)')
    ax[0].set_xlabel('Epohe (Drevja)')
    ax[0].set_ylabel('Izguba (Loss)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Graf 2: Accuracy (XGBoost meri 'error', zato izračunamo 1 - error)
    ucna_točnost = [1 - x for x in rezultati['validation_0']['error']]
    val_točnost = [1 - x for x in rezultati['validation_1']['error']]
    ax[1].plot(x_os, ucna_točnost, label='Učna množica', color='#2196f3')
    ax[1].plot(x_os, val_točnost, label='Validacijska množica', color='#00796b')
    ax[1].set_title('Natančnost (Accuracy)')
    ax[1].set_xlabel('Epohe (Drevja)')
    ax[1].set_ylabel('Natančnost')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Graf 3: Confusion Matrix
    cm = confusion_matrix(y_val, preds_val)
    prikaz_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ni policaja', 'Ležeči policaj'])
    prikaz_cm.plot(ax=ax[2], cmap=plt.cm.BuGn, values_format='d', colorbar=False)
    ax[2].set_title('Matrika zmede (Confusion Matrix)')

    plt.tight_layout()
    plt.savefig(GRAF_IZHOD, dpi=300)
    print(f"Grafi so shranjeni kot slika: '{GRAF_IZHOD}'")
    plt.show()

if __name__ == "__main__":
    treniraj_in_shrani_policaje()
