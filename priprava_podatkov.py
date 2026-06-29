import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import sis1 # Uvozimo tvojo skripto za dekodiranje


MAPA_KATEGORIJE = {
    'dobra_cesta': 0,
    'lezeci': 0,
    'luknja': 0,
    'slaba_cesta': 1
}

TESTNA_DATOTEKA = 'dolgi.BIN'
IZHODNA_MAPA = 'data_out'
OKNO_SEKUNDE = 5.0  

if not os.path.exists(IZHODNA_MAPA):
    os.makedirs(IZHODNA_MAPA)


def izracunaj_znacilke_signala(okno_vzorcev, predpona=""):
    """
    Izračuna matematične lastnosti (značilke) nad N x 3 Numpy matriko.
    """
    znacilke = {}
    osi = ['X', 'Y', 'Z']
    
    if len(okno_vzorcev) == 0:
        return {}

    for i_os, oznaka_osi in enumerate(osi):
        podatki_osi = okno_vzorcev[:, i_os]
        
        polno_ime = f"{predpona}_{oznaka_osi}"
        
        znacilke[f"{polno_ime}_mean"] = np.mean(podatki_osi)
        znacilke[f"{polno_ime}_std"] = np.std(podatki_osi)
        znacilke[f"{polno_ime}_min"] = np.min(podatki_osi)
        znacilke[f"{polno_ime}_max"] = np.max(podatki_osi)
        znacilke[f"{polno_ime}_rms"] = np.sqrt(np.mean(podatki_osi**2))
        znacilke[f"{polno_ime}_ptp"] = np.ptp(podatki_osi)
        znacilke[f"{polno_ime}_kurtosis"] = kurtosis(podatki_osi, fisher=True)
        znacilke[f"{polno_ime}_skew"] = skew(podatki_osi)
        
    return znacilke

def obdelaj_datoteko(filepath, kategorija=None, oznaka=None, samo_gyro=False):
    """
    Dekodira BIN in razreže na 5s okna.
    Če je samo_gyro=True, izračuna in vrne samo značilke za žiroskop.
    """
    print(f"Obdelujem: {filepath}")
    vsi_paketi = sis1.dekodiraj_datoteko(filepath)
    
    paketi_gyro = [p for p in vsi_paketi if p.id == 1]
    fvz_g, sig_g = sis1.sestavi_podatke(paketi_gyro)
    
    if len(sig_g) == 0:
        print(f"  --> Preskakujem datoteko (ni podatkov za gyro).")
        return []


    if not samo_gyro:
        paketi_accel = [p for p in vsi_paketi if p.id == 2]
        fvz_a, sig_a = sis1.sestavi_podatke(paketi_accel)
        if len(sig_a) == 0:
            print(f"  --> Preskakujem datoteko (ni podatkov za accel, ki so tukaj nujne).")
            return []
            
        st_vzorcev_a_5s = int(fvz_a * OKNO_SEKUNDE)
        obdobja_a = len(sig_a) // st_vzorcev_a_5s
    else:

        obdobja_a = float('inf') 

    st_vzorcev_g_5s = int(fvz_g * OKNO_SEKUNDE)
    obdobja_g = len(sig_g) // st_vzorcev_g_5s
    
    stevilo_oken = min(obdobja_g, obdobja_a) if not samo_gyro else obdobja_g
    zbrane_vrstice = []
    
    for i in range(stevilo_oken):

        zacetek_g = i * st_vzorcev_g_5s
        konec_g = (i + 1) * st_vzorcev_g_5s
        okno_gyro = sig_g[zacetek_g:konec_g]
        znac_gyro = izracunaj_znacilke_signala(okno_gyro, predpona="Gyro")
        
        znac_accel = {}
        if not samo_gyro:
            zacetek_a = i * st_vzorcev_a_5s
            konec_a = (i + 1) * st_vzorcev_a_5s
            okno_accel = sig_a[zacetek_a:konec_a]
            znac_accel = izracunaj_znacilke_signala(okno_accel, predpona="Accel")
        
        vrstica = {**znac_accel, **znac_gyro}
        
        vrstica['casovno_okno_sekunde'] = i * OKNO_SEKUNDE
        if kategorija is not None:
            vrstica['izvorna_kategorija'] = kategorija
            vrstica['oznaka_slabo'] = oznaka 
            
        zbrane_vrstice.append(vrstica)
        
    return zbrane_vrstice



if __name__ == "__main__":
    
    # 1. Priprava za učenje (Ohranjene USE OSNOVNE ZNAČILKE - Accel + Gyro)
    print("--- ZAGON USTVARJANJA UČNEGA DATASETA (V2) ---")
    ucni_podatki = []
    
    for mapa_kat, binarna_oznaka in MAPA_KATEGORIJE.items():
        if not os.path.exists(mapa_kat):
            print(f"Opozorilo: Mapa '{mapa_kat}' ne obstaja. Preveri strukturo.")
            continue
            
        for filename in glob.glob(os.path.join(mapa_kat, "*.BIN")):
            vrstice = obdelaj_datoteko(filename, kategorija=mapa_kat, oznaka=binarna_oznaka, samo_gyro=False)
            ucni_podatki.extend(vrstice)
            
    if ucni_podatki:
        df_ucenje = pd.DataFrame(ucni_podatki)
        pot_ucnega = os.path.join(IZHODNA_MAPA, 'train_dataset1.csv')
        df_ucenje.to_csv(pot_ucnega, index=False)
        print(f"\nUSPEH! Dataset za učenje (XGBoost) uspešno shranjen v: {pot_ucnega}")
    else:
        print("\nNapaka: Ni bilo najdenih datotek za učenje.")

    # 2. Priprava testne datoteke - SAMO GYRO!
    print("\n--- ZAGON OBDELAVE TESTNE VOŽNJE (dolgi.BIN) - SAMO GYRO ---")
    if os.path.exists(TESTNA_DATOTEKA):
        test_vrstice = obdelaj_datoteko(TESTNA_DATOTEKA, samo_gyro=True)
        
        if test_vrstice:
            df_test = pd.DataFrame(test_vrstice)
            pot_testnega = os.path.join(IZHODNA_MAPA, 'inference_dolgi_gyro_only.BIN.csv')
            df_test.to_csv(pot_testnega, index=False)
            print(f"USPEH! Podatki testne vožnje (SAMO GYRO) shranjeni v: {pot_testnega}")
            print(f"Datoteka vsebuje samo {len(df_test.columns)-1} stolpcev z značilkami žiroskopa.")
    else:
        print(f"Datoteke {TESTNA_DATOTEKA} ni bilo mogoče najti.")

    print("\nVerzija 2 uspešno zaključena!")

