import os
import datetime
import pandas as pd
import folium
import gpxpy
import joblib


from procesiraj_voznjo import procesiraj_vse


BIN_DATOTEKA = "test_drive.BIN"
GPX_FILE = "testna_voznja_10min.gpx"
ZEMLJEVID_IZHOD = "out.html"

MODEL_POLICAJI = "model_policaji1.pkl"
MODEL_CESTA = "model_ceste1.pkl"

KOREKCIJSKI_FAKTOR = 0.71

def poisci_policaje(csv_pot, model_pot):
    print("\nIskanje policajev")
    paket = joblib.load(model_pot)
    model = paket['model']
    feature_cols = paket['features']
    
    df_test = pd.read_csv(csv_pot)
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

    print(f"Skupaj zaznav: {len(detektirani_casi)}")
    print(f"Po zdruzevanju: {len(zdruzeni_policaji)}")
    return zdruzeni_policaji


def analiziraj_cesto(csv_pot, model_pot):
    print("\nZaznava kvalitete")
    paket = joblib.load(model_pot)
    model = paket['model']
    feature_cols = paket['features']
    
    df_test = pd.read_csv(csv_pot)
    X_test = df_test[feature_cols]
    

    PRAG_SLABO = 0.3
    verjetnosti = model.predict_proba(X_test)[:, 1]
    df_test['predikcija_slabo'] = (verjetnosti >= PRAG_SLABO).astype(int)
    

    df_test['cas_korigiran'] = df_test['casovno_okno_sekunde'] * KOREKCIJSKI_FAKTOR
    
    print(f"Ocenjenih oken za cesto: {len(df_test)}")
    return df_test

def ustvari_interaktivni_zemljevid(gpx_pot, policaji_sekunde, df_cesta):
    print(f"\nUporabljen kor. faktor {KOREKCIJSKI_FAKTOR}")
    
    with open(gpx_pot, "r") as f:
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
    avg_lat = sum(pt.latitude for pt in gpx_tocke) / len(gpx_tocke)
    avg_lon = sum(pt.longitude for pt in gpx_tocke) / len(gpx_tocke)
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles="CartoDB dark_matter")

    sloja_cesta = folium.FeatureGroup(name="Kvaliteta ceste (Zelena=OK, Oranžna=Slabo)")
    sloja_policaji = folium.FeatureGroup(name="Ležeči policaji")
    

    for i in range(len(gpx_tocke) - 1):
        tocka1 = gpx_tocke[i]
        tocka2 = gpx_tocke[i + 1]
        sekunde_od_zacetka = (tocka1.time - T0).total_seconds()
        

        idx_najblizji = (df_cesta['cas_korigiran'] - sekunde_od_zacetka).abs().idxmin()
        oznaka_slabo = df_cesta.loc[idx_najblizji, 'predikcija_slabo']
        
        barva = 'orange' if oznaka_slabo == 1 else 'green'
        
        folium.PolyLine(
            locations=[[tocka1.latitude, tocka1.longitude], [tocka2.latitude, tocka2.longitude]],
            color=barva,
            weight=6,
            opacity=0.8,
            popup=f"Čas GPS: {sekunde_od_zacetka:.1f}s\nStanje: {'Slaba pot' if oznaka_slabo == 1 else 'Dobra pot'}"
        ).add_to(sloja_cesta)
        

    for relativna_sekunda in policaji_sekunde:
        korigirana_sekunda = relativna_sekunda * KOREKCIJSKI_FAKTOR
        absolutni_cas_udarca = T0 + datetime.timedelta(seconds=korigirana_sekunda)
        
        najboljsa_tocka = None
        min_razlika = float('inf')
        for pt in gpx_tocke:
            razlika = abs((pt.time - absolutni_cas_udarca).total_seconds())
            if razlika < min_razlika:
                min_razlika = razlika
                najboljsa_tocka = pt
                
        if najboljsa_tocka is not None:
            folium.Marker(
                location=[najboljsa_tocka.latitude, najboljsa_tocka.longitude],
                popup=f"Ležeči policaj<br>STM ČAS: {round(relativna_sekunda,1)} s",
                icon=folium.Icon(color="red", icon="warning-sign")
            ).add_to(sloja_policaji)
            
            folium.CircleMarker(
                location=[najboljsa_tocka.latitude, najboljsa_tocka.longitude], 
                radius=15, color='red', fill=True, fillOpacity=0.4
            ).add_to(sloja_policaji)


    folium.Marker(
        location=[gpx_tocke[0].latitude, gpx_tocke[0].longitude],
        popup="START VOŽNJE",
        icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)


    sloja_cesta.add_to(m)
    sloja_policaji.add_to(m)
    

    folium.LayerControl(collapsed=False).add_to(m)

    design_html = """
    <style>
        .leaflet-control-layers-expanded {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
            color: #e0f7fa !important;
            border: 1px solid #4db6ac !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.6) !important;
            padding: 10px !important;
        }
        
        .leaflet-control-layers-overlays label {
            color: #80cbc4 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
            margin-top: 5px;
            font-size: 14px;
        }

        .leaflet-control-layers-base {
            display: none !important;
        }
        .leaflet-control-layers-separator {
            display: none !important;
        }

        .header-title {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #004d40, #00897b);
            color: white;
            padding: 12px 35px;
            border-radius: 30px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 1px;
            z-index: 9999;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7);
            border: 1px solid #4db6ac;
            pointer-events: none; 
        }


        .corner-subtitle {
            position: fixed;
            bottom: 25px;
            right: 25px;
            background: linear-gradient(135deg, #00251a, #004d40);
            color: #b2dfdb;
            padding: 10px 25px;
            border-radius: 15px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px;
            font-weight: 700;
            z-index: 9999;
            box-shadow: 0 6px 20px rgba(0,0,0,0.6);
            border: 1px solid #00695c;
            pointer-events: none;
        }
    </style>

    <div class="header-title">RoadQualityAI</div>
    <div class="corner-subtitle">Projekt UM FERI</div>
    """


    m.get_root().html.add_child(folium.Element(design_html))


    m.save(ZEMLJEVID_IZHOD)
    print(f"\nZemljevid shranjen: {ZEMLJEVID_IZHOD}")


if __name__ == "__main__":

    csv_policaji, csv_cesta = procesiraj_vse(BIN_DATOTEKA)
    

    najdeni_policaji = poisci_policaje(csv_policaji, MODEL_POLICAJI)
    cesta_rezultati = analiziraj_cesto(csv_cesta, MODEL_CESTA)
    
    ustvari_interaktivni_zemljevid(GPX_FILE, najdeni_policaji, cesta_rezultati)
