import numpy as np
import struct
import matplotlib.pyplot as plt

MERILO_ZIROSKOP = 8.75e-3
MERILO_POSPESKOMETER = 6.125e-5
MERILO_MAGNETOMETER = 1.5e-3 

ZLOGI_NA_VZOREC = {
    1: 6,
    2: 6,
    3: 6
}

class Paket:
    def __init__(self, id, casovni_zig, podatki):
        self.id = id
        self.casovni_zig = casovni_zig
        self.podatki = podatki

def odstrani_escape_zloge(podatki):
    rezultat = []
    i = 0

    while i < len(podatki):
        if podatki[i] == 0xFE:
            i += 1
            if i < len(podatki):
                rezultat.append(0xFE ^ podatki[i])
        else:
            rezultat.append(podatki[i])
        i += 1

    return bytes(rezultat)

def razcleni_paket(paket_bajti):
    try:
        koristni_podatki_escape = paket_bajti[3:]
        koristni_podatki = odstrani_escape_zloge(koristni_podatki_escape)

        if len(koristni_podatki) < 8:
            return None

        casovni_zig = struct.unpack('<I', koristni_podatki[0:4])[0]

        pozicija = 6
        deli = []

        while pozicija < len(koristni_podatki) - 2:
            id_dela = koristni_podatki[pozicija]

            velikost_dela = struct.unpack(
                '<H',
                koristni_podatki[pozicija+1:pozicija+3]
            )[0] + 1

            zacetek_podatkov = pozicija + 4
            konec_podatkov = zacetek_podatkov + velikost_dela

            if konec_podatkov > len(koristni_podatki):
                break

            podatki_dela = koristni_podatki[zacetek_podatkov:konec_podatkov]

            deli.append((id_dela, podatki_dela))

            pozicija = konec_podatkov

        return casovni_zig, deli

    except:
        return None

def dekodiraj_pakete(podatki):
    paketi = []

    i = 0

    while i < len(podatki) - 1:
        if podatki[i] == 0xFF and podatki[i+1] == 0xFF:
            zacetek = i

            j = i + 2

            while j < len(podatki) - 1:
                if podatki[j] == 0xFF and podatki[j+1] == 0xFF:
                    break
                j += 1

            paket_bajti = podatki[zacetek:j]

            razclenjen_paket = razcleni_paket(paket_bajti)

            if razclenjen_paket:
                casovni_zig, deli = razclenjen_paket

                for id_dela, podatki_dela in deli:
                    paketi.append(
                        Paket(id_dela, casovni_zig, podatki_dela)
                    )

            i = j

        else:
            i += 1

    return paketi

def sestavi_podatke(seznam_paketov):
    rezultati = {}

    ids = set()

    for paket in seznam_paketov:
        ids.add(paket.id)

    for ciljni_id in ids:

        paketi = []

        for paket in seznam_paketov:
            if paket.id == ciljni_id:
                paketi.append(paket)

        if len(paketi) < 2:
            continue

        razlike_casa = []

        for i in range(1, len(paketi)):
            razlika = paketi[i].casovni_zig - paketi[i-1].casovni_zig

            if razlika > 0:
                razlike_casa.append(razlika)

        skupni_cas = 0.0
        stevilo_casov = 0

        for vrednost in razlike_casa:
            skupni_cas += vrednost
            stevilo_casov += 1

        if stevilo_casov == 0:
            continue

        povprecni_cas_paketa = (
            skupni_cas / stevilo_casov
        ) / 1000.0

        zlogi_na_vzorec = ZLOGI_NA_VZOREC.get(ciljni_id, 6)

        vsi_vzorci = []
        seznam_vzorcev = []

        for paket in paketi:

            stevilo_vzorcev = len(paket.podatki) // zlogi_na_vzorec

            seznam_vzorcev.append(stevilo_vzorcev)

            i = 0

            while i + zlogi_na_vzorec <= len(paket.podatki):

                if zlogi_na_vzorec == 6:
                    vzorec = struct.unpack(
                        '<hhh',
                        paket.podatki[i:i+6]
                    )

                    vsi_vzorci.append(vzorec)

                i += zlogi_na_vzorec

        skupno_vzorcev = 0.0
        stevilo_vrednosti = 0

        for vrednost in seznam_vzorcev:
            skupno_vzorcev += vrednost
            stevilo_vrednosti += 1

        if stevilo_vrednosti == 0 or povprecni_cas_paketa == 0:
            continue

        povprecno_stevilo_vzorcev = (
            skupno_vzorcev / stevilo_vrednosti
        )

        frekvenca_vzorcenja = (
            povprecno_stevilo_vzorcev / povprecni_cas_paketa
        )

        matrika = np.array(vsi_vzorci, dtype=float)

        if matrika.size > 0:

            if ciljni_id == 1:
                matrika *= MERILO_ZIROSKOP

            elif ciljni_id == 2:
                matrika *= MERILO_POSPESKOMETER

            elif ciljni_id == 3:
                matrika *= MERILO_MAGNETOMETER

        rezultati[ciljni_id] = (
            frekvenca_vzorcenja,
            matrika
        )

    return rezultati

def prikazi_signal(
    signal,
    naslov="",
    zacetni_indeks=None,
    koncni_indeks=None
):

    if signal.size == 0:
        return

    if zacetni_indeks is None:
        zacetni_indeks = 0

    if koncni_indeks is None:
        koncni_indeks = len(signal)

    signal = signal[zacetni_indeks:koncni_indeks]

    plt.plot(signal[:, 0], label="X")
    plt.plot(signal[:, 1], label="Y")
    plt.plot(signal[:, 2], label="Z")

    plt.title(naslov)

    plt.xlabel("Vzorec")
    plt.ylabel("Vrednost")

    plt.legend()
    plt.grid(True)

def main():

    try:
        with open("luknja_3.BIN", "rb") as datoteka:
            surovi_podatki = datoteka.read()

    except FileNotFoundError:
        print("Napaka: datoteka ni bila najdena!")
        return

    paketi = dekodiraj_pakete(surovi_podatki)

    rezultati = sestavi_podatke(paketi)

    for id_senzorja, (
        frekvenca_vzorcenja,
        signal
    ) in rezultati.items():

        plt.figure()

        prikazi_signal(
            signal,
            f"Signal ID {id_senzorja} "
            f"(Fvz = {frekvenca_vzorcenja:.2f} Hz)"
        )

    for id_senzorja, (
        frekvenca_vzorcenja,
        signal
    ) in rezultati.items():

        if frekvenca_vzorcenja > 0:

            zacetek = int(2 * frekvenca_vzorcenja)
            konec = int(5 * frekvenca_vzorcenja)

            plt.figure()

            prikazi_signal(
                signal,
                f"Interval ID {id_senzorja} (2-5s)",
                zacetek,
                konec
            )

    plt.show()

if __name__ == "__main__":
    main()