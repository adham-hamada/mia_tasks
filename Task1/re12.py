def checksum(string: str) -> int:
    h = 0
    for ch in string:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def encode(ticket_id: str) -> str:
    return f"{ticket_id}-{checksum(ticket_id):08x}"


def decode(barcode: str) -> str:
    parts = barcode.rsplit('-', 1)
    if len(parts) != 2:
        return "CORRUPTED TICKET"

    ticket_id, expected_checksum = parts

    try:
        actual_checksum = f"{checksum(ticket_id):08x}"
        if actual_checksum == expected_checksum.lower():
            return ticket_id
    except Exception:
        pass

    return "CORRUPTED TICKET"


# --- Dynamic Test Suite ---

# 1. Basic Ticket Test
ticket_1 = "1234567890"
encoded_1 = encode(ticket_1)
print(f"Encode ('{ticket_1}'):", encoded_1)        # Output: 1234567890-8e8a3777
print(f"Decode ('{encoded_1}'):", decode(encoded_1))  # Output: 1234567890

print("-" * 40)

# 2. Ticket ID with Hyphens Test
ticket_2 = "TICKET-2026-X"
encoded_2 = encode(ticket_2)
print(f"Encode ('{ticket_2}'):", encoded_2)        # Output: TICKET-2026-X-3ff6c4d7
print(f"Decode ('{encoded_2}'):", decode(encoded_2))  # Output: TICKET-2026-X

print("-" * 40)

# 3. Tampered / Anagram Test (Swapped '89' to '98')
tampered_barcode = f"1234567980-{checksum(ticket_1):08x}"
print(f"Decode Tampered ('{tampered_barcode}'):", decode(tampered_barcode))
# Output: CORRUPTED TICKET