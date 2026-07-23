"""
- Algorithm: 32-Bit Polynomial Rolling Hash (Base-31)

- Formula: hash = (hash * 31 + ASCII_val) & 0xFFFFFFFF

- Multiplying by 31 ensures positional sensitivity—swapping 
adjacent characters (e.g., '12' vs '21') produces wildly different hashes

- (& 0xFFFFFFFF): Simulates 32-bit unsigned integer overflow to keep the output within a fixed 32-bit boundary
"""

class TicketCodec():

    #computing checksum value
    def checksum(self , ticket_id):
        h = 0
        for ch in ticket_id:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h

    def encode(self,ticket_id: str) -> str:
        return f"{ticket_id}-{self.checksum(ticket_id):08x}"

    def decode(self,barcode: str) -> str:
        parts = barcode.rsplit('-', 1)
        if len(parts) != 2:
            return "CORRUPTED TICKET"

        ticket_id, expected_checksum = parts

        try:
            actual_checksum = f"{self.checksum(ticket_id):08x}"
            if actual_checksum == expected_checksum.lower():
                return ticket_id
        except Exception:
            pass

        return "CORRUPTED TICKET"

print(TicketCodec().encode("MIA2026GATE7"))
print(TicketCodec().decode("MIA2026GATE7-f1062085")) #valid

print(TicketCodec().decode("MIA2026GATE4-f1062085")) #tampered