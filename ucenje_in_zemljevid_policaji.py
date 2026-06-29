import os
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import gpxpy
import folium

# --- NASTAVITVE ---
TRAIN_CSV = "train_dataset.csv"
TEST_CSV = "inference_dolgi2.BIN.csv"
GPX_FILE = "testna_voznja_10min.gpx"
ZEMLJEVID_IZHOD = "zemljevid_policajev.html"

def treniraj_model():
    print("--- 1. UČENJE MODELA ---")
    df = pd.read_csv(TRAIN_CSV)
    
    vse_kolone = df.columns.tolist()
    ne_znacilke = ['target', 'cas_okna_start']
    feature_cols = [c for c in vse_kolone if c not in ne_znacilke]
    
    X = df[feature_cols]
    y = df['target']
    
    st_negativnih = len(y[y == 0])
    st_pozitivnih = len(y[y == 1])
    razmerje = st_negativnih / st_pozitivnih if st_pozitivnih > 0 else 1.0
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=razmerje,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print("Ocena modela na validacijski množici:")
    preds_val = model.predict(X_val)
    print(classification_report(y_val, preds_val))
    
    return model, feature_cols

def isci_policaje_v_dolgi_voznji(model, feature_cols):
    print("\n--- 2. ANALIZA DOLGE VOŽNJE ---")
    df_test = pd.read_csv(TEST_CSV)
    
    X_test = df_test[feature_cols]
    casi_oken = df_test['cas_okna_start'].values
    
    verjetnosti = model.predict_proba(X_test)[:, 1] 
    
    meja_zaupanja = 0.72
    zaznave_maska = verjetnosti > meja_zaupanja
    
    detektirani_casi = casi_oken[zaznave_maska]
    
    zdruzeni_policaji = []
    
    if len(detektirani_casi) > 0:
        trenutna_grupa = [detektirani_casi[0]]
        for t in detektirani_casi[1:]:
            if t - trenutna_grupa[-1] <= 6.0: 
                trenutna_grupa.append(t)
            else:
                sredinski_cas = sum(trenutna_grupa) / len(trenutna_grupa)
                zdruzeni_policaji.append(sredinski_cas)
                trenutna_grupa = [t]
        
        sredinski_cas = sum(trenutna_grupa) / len(trenutna_grupa)
        zdruzeni_policaji.append(sredinski_cas)


    zdruzeni_policaji = [t + 2.5 for t in zdruzeni_policaji]

    print(f"Najdenih SUROVIH zaznav (oken): {len(detektirani_casi)}")
    print(f"Unikatnih ležečih policajev po združevanju: {len(zdruzeni_policaji)}")
    print(f"Sekunde udarcev na meritvi (relativno na pot): {[round(t,1) for t in zdruzeni_policaji]}")
    
    return zdruzeni_policaji

def nalozi_in_mapiraj_gpx(detekcije_sekunde, korekcijski_faktor=0.71):
    """
    Parametri:
    - detekcije_sekunde: seznam relativnih sekund, kdaj so detektirani policaji po logiki senzorja.
    - korekcijski_faktor: Vrednost, s katero pomnožimo čas STM32, da ustreza GPX času.
                          < 1.0 pomeni "skrči razdaljo" časa med njimi,
                          > 1.0 pomeni "raztegni razdaljo" časa.
    """
    print(f"\n--- 3. SINHRONIZACIJA Z GPS IN ZEMLJEVID (Faktor {korekcijski_faktor}) ---")
    with open(GPX_FILE, "r") as f:
        gpx = gpxpy.parse(f)
        
    gpx_tocke = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                gpx_tocke.append(pt)
                
    if not gpx_tocke:
         print("GPX datoteka nima točk!")
         return
         
    T0 = gpx_tocke[0].time
    
    koordinate_policajev = []
    
    for relativna_sekunda in detekcije_sekunde:
        
        korigirana_sekunda = relativna_sekunda * korekcijski_faktor
        
        absolutni_cas_udarca = T0 + datetime.timedelta(seconds=korigirana_sekunda)
        
        najboljsa_tocka = None
        min_razlika = float('inf')
        for pt in gpx_tocke:
            razlika = abs((pt.time - absolutni_cas_udarca).total_seconds())
            if razlika < min_razlika:
                min_razlika = razlika
                najboljsa_tocka = pt
                
        if najboljsa_tocka is not None:
            koordinate_policajev.append((najboljsa_tocka.latitude, najboljsa_tocka.longitude, relativna_sekunda))

    avg_lat = sum(pt.latitude for pt in gpx_tocke) / len(gpx_tocke)
    avg_lon = sum(pt.longitude for pt in gpx_tocke) / len(gpx_tocke)
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15)
    
    linija = [(pt.latitude, pt.longitude) for pt in gpx_tocke]
    folium.PolyLine(locations=linija, color="blue", weight=4, opacity=0.7).add_to(m)
    
    folium.Marker(
        location=[gpx_tocke[0].latitude, gpx_tocke[0].longitude],
        popup="START VOŽNJE",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    
    for idx, (lat, lon, sek) in enumerate(koordinate_policajev):
        oznaka = folium.Marker(
            location=[lat, lon],
            popup=f"Ležeči policaj #{idx+1}<br>STM ČAS: {round(sek,1)} s",
            icon=folium.Icon(color="red", icon="warning-sign")
        )
        oznaka.add_to(m)
        folium.CircleMarker(
            location=[lat, lon], radius=15, color='red', fill=True, fillOpacity=0.4
        ).add_to(m)

    m.save(ZEMLJEVID_IZHOD)
    print(f"\nUSPEH! Preveri svoj brskalnik. Zemljevid je v datoteki: {ZEMLJEVID_IZHOD}")


if __name__ == "__main__":
    if not os.path.exists(TRAIN_CSV):
         print("NAPAKA: Manjka train_dataset.csv! Najprej poženi prvo skripto.")
    else:
         model, atributi = treniraj_model()
         casovne_znacke = isci_policaje_v_dolgi_voznji(model, atributi)
         
         nalozi_in_mapiraj_gpx(casovne_znacke, korekcijski_faktor=0.71) 


