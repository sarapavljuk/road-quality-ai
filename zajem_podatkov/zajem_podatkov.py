import serial
import time

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200

counters = {
    "dobra_cesta": 1,
    "slaba_cesta": 1,
    "lezeci": 1,
    "dolgi": 1
}


def record_sequence(label, duration):
    file_name = f"{label}{counters[label]}.BIN"

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception:
        print(f"\nNAPAKA: Ni mogoče odpreti vrat {SERIAL_PORT}.")
        print("Preveri povezavo in zapri ostale programe.")
        return

    try:
        print(f"\nDatoteka: {file_name}")
        input("Pritisni ENTER za začetek snemanja...")

        ser.reset_input_buffer()

        with open(file_name, "wb") as f:
            print(f" >>> SNEMAM ({duration}s)... ", end="", flush=True)

            start_time = time.time()
            while (time.time() - start_time) < duration:
                if ser.in_waiting > 0:
                    raw_data = ser.read(ser.in_waiting)
                    f.write(raw_data)

        print("KONČANO!")
        print(f"Podatki so shranjeni v datoteko {file_name}.")

        counters[label] += 1

    finally:
        ser.close()


def get_duration():
    try:
        duration = float(input("Vnesi dolžino posnetka v sekundah: ").strip())
        if duration <= 0:
            print("Dolžina mora biti večja od 0.")
            return None
        return duration
    except ValueError:
        print("Napačen vnos za dolžino posnetka.")
        return None


def main():
    while True:
        print("\n" + "=" * 50)
        print(" ZAJEM PODATKOV")
        print("=" * 50)
        print("1 - dobra cesta")
        print("2 - slaba cesta")
        print("3 - ležeči policaj")
        print("4 - dolgi posnetek")
        print("5 - izhod iz programa")

        choice = input("Izberi opcijo: ").strip()

        if choice == "1":
            duration = get_duration()
            if duration is not None:
                record_sequence("dobra_cesta", duration)

        elif choice == "2":
            duration = get_duration()
            if duration is not None:
                record_sequence("slaba_cesta", duration)

        elif choice == "3":
            duration = get_duration()
            if duration is not None:
                record_sequence("lezeci", duration)
        
        elif choice == "4":
            duration = get_duration()
            if duration is not None:
                record_sequence("dolgi", duration)

        elif choice == "5":
            print("Izhod iz programa.")
            break

        else:
            print("Neveljavna izbira. Poskusi znova.")


if __name__ == "__main__":
    main()