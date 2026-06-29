import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# --- KONFIGURACIJA ---
UCNI_PODATKI_POT = 'data_out/train_dataset_original.csv'
MODEL_IZHOD = 'model_ceste.pkl'
GRAF_IZHOD = "grafi_ceste.png"

def treniraj_in_shrani_ceste():
    print("--- 1. UČENJE MODELA ZA KAKOVOST CESTE ---")
    
    try:
        df_train = pd.read_csv(UCNI_PODATKI_POT)
    except FileNotFoundError:
        print(f"NAPAKA: Datoteka {UCNI_PODATKI_POT} ne obstaja!")
        return

    ne_znacilke = ['casovno_okno_sekunde', 'izvorna_kategorija', 'oznaka_slabo']
    feature_cols = [col for col in df_train.columns if col not in ne_znacilke and 'Gyro' in col]

    X = df_train[feature_cols]
    y = df_train['oznaka_slabo']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"   Model se uči na {len(feature_cols)} značilkah...")

    # POPRAVKI ZA OVERFITTING (max_depth=3, learning_rate, subsample, colsample_bytree, early_stopping)
    model = xgb.XGBClassifier(
        random_state=42, 
        n_estimators=100, 
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    print("\n--- 2. OCENA MODELA ---")
    val_pred = model.predict(X_val)
    natančnost = accuracy_score(y_val, val_pred)
    print(f"   Najboljša iteracija: {model.best_iteration}")
    print(f"   Natančnost modela na validacijski množici: {natančnost*100:.2f}%")
    print("\n Poročilo klasifikacije:\n", classification_report(y_val, val_pred))

    paket_za_shranjevanje = {
        'model': model,
        'features': feature_cols
    }
    joblib.dump(paket_za_shranjevanje, MODEL_IZHOD)
    print(f"Model uspešno shranjen v '{MODEL_IZHOD}'.")

    # --- 3. IZRIS GRAFOV ---
    print("\n--- 3. GENERIRANJE GRAFOV ---")
    rezultati = model.evals_result()
    epochs = len(rezultati['validation_0']['logloss'])
    x_os = range(0, epochs)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RoadQualityAi - Evalvacija modela za Kakovost ceste', fontsize=16, fontweight='bold', color='#2196f3')

    # Graf 1: Log Loss
    ax[0].plot(x_os, rezultati['validation_0']['logloss'], label='Učna množica', color='#2196f3')
    ax[0].plot(x_os, rezultati['validation_1']['logloss'], label='Validacijska množica', color='#00796b')
    ax[0].set_title('Krivulja izgube (Log Loss)')
    ax[0].set_xlabel('Epohe (Drevja)')
    ax[0].set_ylabel('Izguba (Loss)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Graf 2: Accuracy
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
    cm = confusion_matrix(y_val, val_pred)
    prikaz_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dobra cesta', 'Slaba cesta'])
    prikaz_cm.plot(ax=ax[2], cmap=plt.cm.Blues, values_format='d', colorbar=False)
    ax[2].set_title('Matrika zmede (Confusion Matrix)')

    plt.tight_layout()
    plt.savefig(GRAF_IZHOD, dpi=300)
    print(f"Grafi so shranjeni kot slika: '{GRAF_IZHOD}'")
    plt.show()

if __name__ == "__main__":
    treniraj_in_shrani_ceste()
