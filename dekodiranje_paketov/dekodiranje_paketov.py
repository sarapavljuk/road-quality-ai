import struct

class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

def unstuff_bytes(data):
    result = []
    i = 0

    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i < len(data):
                result.append(0xFE ^ data[i])
        else:
            result.append(data[i])
        i += 1

    return bytes(result)

def crc16(data):
    crc = 0xFFFF

    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1

    return crc

def parse_packet(packet_bytes):
    try:
        payload_stuffed = packet_bytes[3:]
        payload = unstuff_bytes(payload_stuffed)

        if len(payload) < 8:
            return None

        timestamp = struct.unpack('<I', payload[0:4])[0]
        packet_size = struct.unpack('<H', payload[4:6])[0] + 1

        received_crc = struct.unpack('<H', payload[-2:])[0]
        computed_crc = crc16(payload[:-2])

        if received_crc != computed_crc:
            return None

        pos = 6
        chunks = []

        while pos < len(payload) - 2:
            if pos + 4 > len(payload):
                break

            chunk_id = payload[pos]
            chunk_size = struct.unpack('<H', payload[pos+1:pos+3])[0] + 1

            data_start = pos + 4
            data_end = data_start + chunk_size

            if data_end > len(payload):
                break

            chunk_data = payload[data_start:data_end]

            chunks.append((chunk_id, chunk_data))

            pos = data_end

        return timestamp, chunks

    except:
        return None

def decode_packets(data):
    paketi = []

    i = 0
    while i < len(data) - 1:
        if data[i] == 0xFF and data[i+1] == 0xFF:
            start = i

            j = i + 2
            while j < len(data) - 1:
                if data[j] == 0xFF and data[j+1] == 0xFF:
                    break
                j += 1

            packet_bytes = data[start:j]

            parsed = parse_packet(packet_bytes)

            if parsed:
                timestamp, chunks = parsed

                for chunk_id, chunk_data in chunks:
                    paketi.append(Paket(chunk_id, timestamp, chunk_data))

            i = j
        else:
            i += 1

    return paketi

def main():
    with open("naloga_1_log.BIN", "rb") as f:
        raw = f.read()

    print("Velikost datoteke:", len(raw))

    paketi = decode_packets(raw)

    print("Število paketov:", len(paketi))

if __name__ == "__main__":
    main()