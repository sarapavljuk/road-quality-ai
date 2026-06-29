import pandas as pd
import xgboost as xgb
import xml.etree.ElementTree as ET
from datetime import datetime
import folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# --- KONFIGURACIJA ---
UCNI_PODATKI_POT = 'data_out/train_dataset_original.csv'
TESTNI_PODATKI_POT = 'data_out/inference_dolgi_gyro_only.BIN.csv'
GPX_POT = 'test.gpx'
ZEMLJEVID_IZHOD = 'rezultat_ceste_zemljevid.html'
KOREKCIJSKI_FAKTOR = 0.71   # Faktor, s katerim pomnožimo čas STM32, da ustreza GPX času
PRAG_SLABO = 0.3

def treniraj_xgboost():
    print("1. Nalaganje podatkov za učenje...")
    df_train = pd.read_csv(UCNI_PODATKI_POT)
    df_test = pd.read_csv(TESTNI_PODATKI_POT)

    znacilke_za_ucenje = [col for col in df_test.columns if col not in ['casovno_okno_sekunde', 'izvorna_kategorija', 'oznaka_slabo']]

    print(f"   Model se bo učil na teh {len(znacilke_za_ucenje)} značilkah: {znacilke_za_ucenje[0]} ... itd.")

    X = df_train[znacilke_za_ucenje]
    y = df_train['oznaka_slabo']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n2. Učenje XGBoost modela...")
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    natančnost = accuracy_score(y_val, val_pred)
    print(f"   Natančnost modela na validacijski množici: {natančnost*100:.2f}%")
    print("\n Poročilo klasifikacije:\n", classification_report(y_val, val_pred))

    print("\n3. Izvajanje predikcije na dolgi.BIN...")
    X_inference = df_test[znacilke_za_ucenje]
    verjetnosti = model.predict_proba(X_inference)[:, 1]
    napovedi = (verjetnosti >= PRAG_SLABO).astype(int)
    
    df_test['predikcija_slabo'] = napovedi
    return df_test

def ekstrahiraj_gpx(pot):
    """Parsira GPX in vrne seznam slovarjev (čas, lat, lon) s preračunanim časom v sekundah od prve točke."""
    print("4. Branje GPX datoteke za lociranje...")
    tree = ET.parse(pot)
    root = tree.getroot()
    ns = {'gpx': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    prefix = '{' + ns['gpx'] + '}' if ns else ''

    gpx_tocke = []
    zacetni_cas_dt = None

    for trkpt in root.iter(prefix + 'trkpt' if prefix else 'trkpt'):
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        time_elem = trkpt.find(prefix + 'time' if prefix else 'time')

        if time_elem is not None:
             time_str = time_elem.text
             if '.' in time_str:
                 time_str = time_str.split('.')[0] + 'Z'
             dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

             if zacetni_cas_dt is None:
                 zacetni_cas_dt = dt

             sekunde_od_zacetka = (dt - zacetni_cas_dt).total_seconds()
             gpx_tocke.append({'sekunde': sekunde_od_zacetka, 'lat': lat, 'lon': lon})

    print(f"   Najdenih {len(gpx_tocke)} točk v GPX.")
    return pd.DataFrame(gpx_tocke)

def ustvari_zemljevid(df_rezultati, df_gpx):
    print("5. Sinhronizacija senzorike z GPS in generacija zemljevida...")

    prva_tocka = df_gpx.iloc[0]
    mapa = folium.Map(location=[prva_tocka['lat'], prva_tocka['lon']], zoom_start=15)

    df_napovedi = df_rezultati[['casovno_okno_sekunde', 'predikcija_slabo']].copy()
    df_napovedi['cas_korigiran'] = df_napovedi['casovno_okno_sekunde'] * KOREKCIJSKI_FAKTOR

    for i in range(len(df_gpx) - 1):
        tocka1 = df_gpx.iloc[i]
        tocka2 = df_gpx.iloc[i + 1]

        idx_najblizji = (df_napovedi['cas_korigiran'] - tocka1['sekunde']).abs().idxmin()
        oznaka_slabo = df_napovedi.loc[idx_najblizji, 'predikcija_slabo']

        barva = 'orange' if oznaka_slabo == 1 else 'green'

        folium.PolyLine(
            locations=[[tocka1['lat'], tocka1['lon']], [tocka2['lat'], tocka2['lon']]],
            color=barva,
            weight=6,
            opacity=0.8,
            popup=f"Čas GPS: {tocka1['sekunde']:.1f}s\nStanje: {'Slaba pot' if oznaka_slabo == 1 else 'Dobra pot'}"
        ).add_to(mapa)

    mapa.save(ZEMLJEVID_IZHOD)
    print(f"\nZEMLJEVID KONČAN! Shranjen je kot: {ZEMLJEVID_IZHOD}")
    print("Odpri datoteko z brskalnikom.")

if __name__ == "__main__":
    rezultati_testne_voznje = treniraj_xgboost()

    if int(len(rezultati_testne_voznje)) > 0:
        gpx_df = ekstrahiraj_gpx(GPX_POT)
        ustvari_zemljevid(rezultati_testne_voznje, gpx_df)
    else:
        print("Napaka - prazen rezultat datotek!")

