"""Motor de codificación y decodificación Nikon NEF 14-bit Lossless (etiqueta 0x4630).

Forma parte de la librería `nefinject`, proporcionando compresión y descompresión de flujo
RAW para sensores Nikon FX/DX (D610, D800, D750) con paridad matemática bit a bit.

Estructura del formato y algoritmos:
- Encabezado Lossless (24 bytes): Define versión del códec, dimensiones de imagen RAW,
  límites de compresión y tabla de frecuencias de Huffman.
- Predicción DPCM: Predicción diferencial por canal de color dependiente de la paridad de fila
  y columna (R, Gr, Gb, B), calculando el residuo de cada sensel respecto a sus predecesores.
- Codificación Huffman Canónica: Mapeo de categorías de amplitud diferencial (0..14) a
  secuencias de bits de longitud variable según el estándar Nikon, seguido de los bits de
  magnitud diferencial codificados en orden MSB (Most Significant Bit first).

Arquitectura de motores intercambiables:
1. Motor Numba JIT (@njit): Rendimiento ultra-rápido (~0.18 s para 24.49 Megapíxeles) cuando
   el compilador JIT Numba está presente en el entorno de ejecución.
2. Motor NumPy Vectorizado Puro: Codificación completa en ~0.85 s sin dependencias externas
   compiladas, empleando matrices de predicción por paridad, tabla LUT de 32,768 entradas
   y empaquetado de bits acumulado mediante histogramas y bincount.
3. Motor Python de referencia (`MSBBitReader` / `MSBBitWriter`): Herramienta diagnóstica
   para decodificación selectiva de filas y validación bit a bit.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .container import LosslessHeader
from .models import CodecError, sha256_bytes, sha256_file

VERSION = "0.5.0"

COUNTS_14_LOSSLESS = (0, 1, 4, 2, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0)
SYMBOLS_14_LOSSLESS = (7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14)

try:
    from numba import njit
except Exception:  # Captura ImportError o incompatibilidad de versión NumPy/Numba
    njit = None


def numba_acceleration_available() -> bool:
    """Devuelve True si el compilador JIT Numba está operativo en el entorno actual."""
    return njit is not None


class CanonicalHuffman:
    """Mapea categorías 0..14 a códigos canónicos Nikon 14-bit Lossless, leyendo MSB primero."""

    def __init__(self) -> None:
        self.encode_map: dict[int, tuple[int, int]] = {}
        self.decode_map: dict[tuple[int, int], int] = {}
        k, code = 0, 0
        for length, count in enumerate(COUNTS_14_LOSSLESS, start=1):
            code <<= 1
            for _ in range(count):
                sym = SYMBOLS_14_LOSSLESS[k]
                self.encode_map[sym] = (code, length)
                self.decode_map[(code, length)] = sym
                code += 1
                k += 1
            if code > 1 << length:
                raise CodecError("Código Huffman no prefijo")
        if k != len(SYMBOLS_14_LOSSLESS) or set(self.encode_map) != set(range(15)):
            raise CodecError("Tabla Huffman incompleta")
        self.max_length = max(length for _, length in self.encode_map.values())

    def symbol(self, r: "MSBBitReader") -> int:
        code = 0
        for ln in range(1, self.max_length + 1):
            code = (code << 1) | r.read(1)
            if (sym := self.decode_map.get((code, ln))) is not None:
                return sym
        raise CodecError("Secuencia Huffman inválida")

    def json_codes(self) -> dict[str, str]:
        return {str(s): format(c, f"0{l}b") for s, (c, l) in sorted(self.encode_map.items())}


# ---------------------------------------------------------------------------
# Precomputación de LUT de 32,768 entradas para diferencias d en [-16383, +16383]
# ---------------------------------------------------------------------------

def _build_diff_lut() -> tuple[np.ndarray, np.ndarray]:
    huff = CanonicalHuffman()
    codes = np.zeros(32768, dtype=np.uint32)
    lengths = np.zeros(32768, dtype=np.uint8)
    for d in range(-16383, 16384):
        idx = d + 16383
        if d == 0:
            c, ln = huff.encode_map[0]
            codes[idx] = c
            lengths[idx] = ln
        else:
            bits = abs(d).bit_length()
            c, ln = huff.encode_map[bits]
            extra = d if d > 0 else d + ((1 << bits) - 1)
            codes[idx] = (c << bits) | extra
            lengths[idx] = ln + bits
    return codes, lengths


_DIFF_CODE_LUT, _DIFF_LEN_LUT = _build_diff_lut()


# ---------------------------------------------------------------------------
# Motor 1: Codificador 100% NumPy Vectorizado (0 bucles sobre píxeles)
# ---------------------------------------------------------------------------

def encode_nikon_lossless_numpy(
    raw: np.ndarray,
    seed_parities: tuple[tuple[int, int], tuple[int, int]],
    *,
    chunk_samples: int = 2_000_000,
) -> tuple[np.ndarray, int]:
    """Codifica un mosaico Bayer 2D uint16 de 14 bits utilizando exclusivamente operaciones
    vectorizadas de NumPy (predictor espacial 2D + LUT de 32,768 entradas + empaquetado
    por bincount en palabras de 32 bits). Produce exactamente el mismo flujo binario
    que el codificador Nikon de referencia y Numba.
    """
    if raw.dtype != np.uint16 or raw.ndim != 2 or raw.shape[1] < 2:
        raise CodecError("Se requiere un ndarray 2D uint16 con ancho >= 2")
    if int(np.max(raw)) >= 1 << 14:
        raise CodecError("Existen muestras mayores que 16383 (fuera de 14 bits)")

    # 1. Predictor diferencial espacial vectorizado por paridad de fila y columna
    pred = np.empty_like(raw, dtype=np.int32)
    pred[:, 2:] = raw[:, :-2]
    pred[0, 0] = seed_parities[0][0]
    pred[0, 1] = seed_parities[0][1]
    if raw.shape[0] > 1:
        pred[1, 0] = seed_parities[1][0]
        pred[1, 1] = seed_parities[1][1]
    if raw.shape[0] > 2:
        pred[2::2, 0] = raw[0:-2:2, 0]
        pred[2::2, 1] = raw[0:-2:2, 1]
    if raw.shape[0] > 3:
        pred[3::2, 0] = raw[1:-2:2, 0]
        pred[3::2, 1] = raw[1:-2:2, 1]

    diff_idx = (raw.astype(np.int32) - pred + 16383).ravel()
    del pred
    if int(diff_idx.min()) < 0 or int(diff_idx.max()) > 32766:
        raise CodecError("Diferencia predictiva fuera de rango [-16383, 16383]")

    # 2. Empaquetado vectorizado MSB-first en palabras de 32 bits por bloques
    n = diff_idx.size
    out_chunks: list[bytes] = []
    carry_word = np.uint64(0)
    carry_bits = 0
    total_bits_accum = 0

    for start in range(0, n, chunk_samples):
        sub = diff_idx[start : start + chunk_samples]
        c = _DIFF_CODE_LUT[sub].astype(np.uint64)
        l = _DIFF_LEN_LUT[sub].astype(np.int64)

        bit_ends = np.cumsum(l) + carry_bits
        total_bits = int(bit_ends[-1])
        total_bits_accum += int(bit_ends[-1]) - carry_bits
        bit_starts = bit_ends - l

        word_idx = (bit_starts >> 5).astype(np.int64)
        bit_in_word = (bit_starts & 31).astype(np.uint64)
        lu = l.astype(np.uint64)
        total_span = bit_in_word + lu

        hi_shift = np.where(total_span <= 32, np.uint64(32) - total_span, np.uint64(0))
        lo_shift = np.where(total_span > 32, total_span - np.uint64(32), np.uint64(0))

        hi_contrib = np.where(total_span <= 32, c << hi_shift, c >> lo_shift)
        lo_contrib = np.where(
            total_span > 32,
            (c << (np.uint64(64) - total_span)) & np.uint64(0xFFFFFFFF),
            np.uint64(0),
        )

        num_words = (total_bits + 31) >> 5
        words = np.bincount(
            word_idx, weights=hi_contrib.astype(np.float64), minlength=num_words + 1
        ).astype(np.int64).astype(np.uint32)

        spill = total_span > 32
        if np.any(spill):
            words_lo = np.bincount(
                word_idx[spill] + 1,
                weights=lo_contrib[spill].astype(np.float64),
                minlength=num_words + 1,
            ).astype(np.int64).astype(np.uint32)
            words |= words_lo

        if carry_bits > 0:
            words[0] |= np.uint32(carry_word)

        full_words = total_bits >> 5
        rem_bits = total_bits & 31
        if full_words > 0:
            out_chunks.append(words[:full_words].astype(">u4").tobytes())
        if rem_bits > 0:
            carry_word = np.uint64(words[full_words])
            carry_bits = rem_bits
        else:
            carry_word = np.uint64(0)
            carry_bits = 0

    if carry_bits > 0:
        n_rem_bytes = (carry_bits + 7) >> 3
        out_chunks.append(int(carry_word).to_bytes(4, "big")[:n_rem_bytes])

    stream_bytes = b"".join(out_chunks)
    return np.frombuffer(stream_bytes, dtype=np.uint8), total_bits_accum


# ---------------------------------------------------------------------------
# Motor 2: Codificador y Decodificador JIT Numba (@njit)
# ---------------------------------------------------------------------------

if njit is not None:
    @njit(cache=True)
    def _numba_encode(
        raw: np.ndarray,
        code_by_category: np.ndarray,
        len_by_category: np.ndarray,
        seed0: int,
        seed1: int,
        seed2: int,
        seed3: int,
    ) -> tuple[np.ndarray, int]:
        h, w = raw.shape
        max_symbol_len = 0
        for k in range(15):
            if len_by_category[k] > max_symbol_len:
                max_symbol_len = len_by_category[k]
        capacity = (h * w * (max_symbol_len + 14) + 7) // 8
        dst = np.zeros(capacity, dtype=np.uint8)
        next_byte = 0
        bitbuf = 0
        pending = 0
        used_bits = 0
        seeds = np.array([[seed0, seed1], [seed2, seed3]], dtype=np.int64)
        for y in range(h):
            horizontal0 = 0
            horizontal1 = 0
            for x in range(w):
                val = int(raw[y, x])
                parity = x & 1
                if x < 2:
                    pred = seeds[y & 1, parity]
                else:
                    pred = horizontal0 if parity == 0 else horizontal1
                diff = val - pred
                mag = abs(diff)
                cat = 0
                while mag:
                    cat += 1
                    mag >>= 1
                c = code_by_category[cat]
                bits = len_by_category[cat]
                bitbuf = (bitbuf << bits) | c
                pending += bits
                used_bits += bits
                while pending >= 8:
                    pending -= 8
                    dst[next_byte] = (bitbuf >> pending) & 255
                    next_byte += 1
                    bitbuf &= (1 << pending) - 1
                if cat:
                    extra = diff if diff > 0 else diff + (1 << cat) - 1
                    bitbuf = (bitbuf << cat) | extra
                    pending += cat
                    used_bits += cat
                    while pending >= 8:
                        pending -= 8
                        dst[next_byte] = (bitbuf >> pending) & 255
                        next_byte += 1
                        bitbuf &= (1 << pending) - 1
                if x < 2:
                    seeds[y & 1, parity] = val
                if parity:
                    horizontal1 = val
                else:
                    horizontal0 = val
        if pending:
            dst[next_byte] = (bitbuf << (8 - pending)) & 255
            next_byte += 1
        return dst[:next_byte], used_bits

    @njit(cache=True)
    def _numba_decode(
        stream: np.ndarray,
        width: int,
        height: int,
        tree_left: np.ndarray,
        tree_right: np.ndarray,
        tree_symbol: np.ndarray,
        seed0: int,
        seed1: int,
        seed2: int,
        seed3: int,
    ) -> tuple[np.ndarray, int]:
        out = np.zeros((height, width), dtype=np.uint16)
        seeds = np.array([[seed0, seed1], [seed2, seed3]], dtype=np.int64)
        bit_position = 0
        total = stream.size * 8
        for y in range(height):
            h0, h1 = 0, 0
            for x in range(width):
                node = 0
                while tree_symbol[node] == -1:
                    if bit_position >= total:
                        return out, -1
                    k = bit_position >> 3
                    shift = 7 - (bit_position & 7)
                    bit = (stream[k] >> shift) & 1
                    bit_position += 1
                    node = tree_right[node] if bit else tree_left[node]
                    if node < 0:
                        return out, -2
                cat = tree_symbol[node]
                diff = 0
                if cat:
                    v = 0
                    for _ in range(cat):
                        if bit_position >= total:
                            return out, -1
                        k = bit_position >> 3
                        shift = 7 - (bit_position & 7)
                        v = (v << 1) | ((stream[k] >> shift) & 1)
                        bit_position += 1
                    diff = v if (v & (1 << (cat - 1))) else v - ((1 << cat) - 1)
                parity = x & 1
                if x < 2:
                    val = seeds[y & 1, parity] + diff
                    seeds[y & 1, parity] = val
                else:
                    val = (h1 if parity else h0) + diff
                if val < 0 or val > 16383:
                    return out, -3
                out[y, x] = val
                if parity:
                    h1 = val
                else:
                    h0 = val
        return out, bit_position


# ---------------------------------------------------------------------------
# Motor 3: Lector/Escritor Bit a Bit en Python Puro (Referencia / Prefijos)
# ---------------------------------------------------------------------------

class MSBBitReader:
    def __init__(self, data: bytes | np.ndarray) -> None:
        self.data = memoryview(data)
        self.byte_index = 0
        self.bits = 0
        self.bitbuf = 0
        self.consumed_bits = 0

    def read(self, count: int) -> int:
        if count == 0:
            return 0
        if count < 0 or count > 24:
            raise CodecError("Longitud de lectura de bits inválida")
        while self.bits < count:
            if self.byte_index >= len(self.data):
                raise CodecError(f"Fin prematuro de flujo Nikon, bit {self.consumed_bits}")
            self.bitbuf = (self.bitbuf << 8) | self.data[self.byte_index]
            self.byte_index += 1
            self.bits += 8
        self.bits -= count
        self.consumed_bits += count
        result = (self.bitbuf >> self.bits) & ((1 << count) - 1)
        self.bitbuf &= (1 << self.bits) - 1
        return result


class MSBBitWriter:
    def __init__(self, fd: BinaryIO) -> None:
        self.fd = fd
        self.buf = bytearray()
        self.pending = 0
        self.bits = 0
        self.total_bits = 0
        self.written_bytes = 0

    def write(self, code: int, bits: int) -> None:
        if bits < 0 or bits > 24 or code < 0 or code >= 1 << bits:
            raise CodecError("Código binario inválido")
        self.pending = (self.pending << bits) | code
        self.bits += bits
        self.total_bits += bits
        while self.bits >= 8:
            self.bits -= 8
            self.buf.append((self.pending >> self.bits) & 0xFF)
            self.pending &= (1 << self.bits) - 1
        if len(self.buf) >= 1_048_576:
            self.fd.write(self.buf)
            self.written_bytes += len(self.buf)
            self.buf.clear()

    def finish(self) -> None:
        if self.bits:
            self.buf.append((self.pending << (8 - self.bits)) & 0xFF)
            self.pending = self.bits = 0
        if self.buf:
            self.fd.write(self.buf)
            self.written_bytes += len(self.buf)
            self.buf.clear()


class NikonLosslessCodec:
    """Fachada unificada de codificación/decodificación Nikon D610 14-bit Lossless (0x4630)."""

    def __init__(self, header: LosslessHeader, *, backend: str = "auto") -> None:
        if not header.supported or header.bit_depth != 14 or header.curve_mode != "identity_0x46":
            raise CodecError("Encabezamiento no admitido por este codificador")
        if backend not in ("auto", "numba", "numpy", "python"):
            raise CodecError(f"Backend de códec desconocido: {backend}")
        self.header = header
        self.backend = backend
        self.huff = CanonicalHuffman()

        enc = self.huff.encode_map
        self.codes = np.array([enc[k][0] for k in range(15)], dtype=np.int64)
        self.lengths = np.array([enc[k][1] for k in range(15)], dtype=np.int64)
        tree = [[-1, -1, -1]]
        for category, (code, length) in enc.items():
            node = 0
            for p in range(length - 1, -1, -1):
                bit = (code >> p) & 1
                next_node = tree[node][bit]
                if next_node < 0:
                    next_node = len(tree)
                    tree[node][bit] = next_node
                    tree.append([-1, -1, -1])
                node = next_node
            tree[node][2] = category
        self.left = np.asarray([t[0] for t in tree], dtype=np.int32)
        self.right = np.asarray([t[1] for t in tree], dtype=np.int32)
        self.symbols = np.asarray([t[2] for t in tree], dtype=np.int32)

    def _seeds(self) -> tuple[int, int, int, int]:
        return tuple(int(v) for pair in self.header.seed_parities for v in pair)  # type: ignore[return-value]

    def encode_in_memory(self, data: np.ndarray) -> tuple[np.ndarray, int, str]:
        """Codifica un array 2D uint16 en memoria y devuelve (bytes_uint8, bits_totales, motor_usado)."""
        raw = np.asarray(data)
        if raw.dtype != np.uint16 or raw.ndim != 2 or raw.shape[1] < 2:
            raise CodecError("Se requiere uint16 2-D")
        if int(np.max(raw)) > 16383:
            raise CodecError("Muestra mayor de 14 bits")

        use_numba = (self.backend == "numba") or (self.backend == "auto" and njit is not None)
        if use_numba:
            if njit is None:
                raise CodecError("Se solicitó backend='numba' pero Numba no está disponible")
            bits_data, bits_used = _numba_encode(raw, self.codes, self.lengths, *self._seeds())
            return bits_data, int(bits_used), "numba"

        if self.backend in ("auto", "numpy"):
            bits_data, bits_used = encode_nikon_lossless_numpy(raw, self.header.seed_parities)
            return bits_data, int(bits_used), "numpy_vectorized"

        # Fallback puro Python
        import io
        buf = io.BytesIO()
        wr = MSBBitWriter(buf)
        seeds = [list(v) for v in self.header.seed_parities]
        for y, row in enumerate(raw):
            horizontal = [0, 0]
            for x, v0 in enumerate(row):
                v = int(v0)
                parity = x & 1
                predictor = seeds[y & 1][parity] if x < 2 else horizontal[parity]
                diff = v - predictor
                if not -16383 <= diff <= 16383:
                    raise CodecError("Diferencia predictiva fuera de rango")
                bits = abs(diff).bit_length()
                code, length = self.huff.encode_map[bits]
                wr.write(code, length)
                if bits:
                    extra = diff if diff > 0 else diff + ((1 << bits) - 1)
                    wr.write(extra, bits)
                if x < 2:
                    seeds[y & 1][parity] = v
                horizontal[parity] = v
        wr.finish()
        return np.frombuffer(buf.getvalue(), dtype=np.uint8), int(wr.total_bits), "python"

    def encode_to_file(self, data: np.ndarray, destination: str | Path) -> dict[str, Any]:
        destination = Path(destination)
        if destination.exists():
            raise CodecError("El archivo .bin de salida existe")
        bits_data, bits_used, engine_used = self.encode_in_memory(data)
        digest = sha256_bytes(bits_data)
        with destination.open("xb") as fd:
            bits_data.tofile(fd)
        return {
            "path": str(destination),
            "sha256": digest,
            "bytes": int(bits_data.size),
            "encoded_bits": int(bits_used),
            "sample_count": int(data.size),
            "bits_per_sample_mean": bits_used / data.size,
            "accelerated_with": engine_used,
        }

    def decode(
        self,
        stream: bytes | np.ndarray,
        width: int,
        rows: int,
        *,
        out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Decodifica el flujo Nikon 14-bit Lossless (usando Numba si está activo, o Python puro)."""
        if width <= 1 or rows <= 0 or width > 100_000 or rows > 100_000:
            raise CodecError("Dimensiones inválidas")

        use_numba = (self.backend != "python") and (njit is not None)
        if use_numba:
            arr_u8 = (
                np.frombuffer(stream, dtype=np.uint8)
                if isinstance(stream, (bytes, bytearray, memoryview))
                else np.asarray(stream, dtype=np.uint8)
            )
            decoded, used = _numba_decode(
                arr_u8, width, rows, self.left, self.right, self.symbols, *self._seeds()
            )
            if used < 0:
                raise CodecError(f"Error Numba al decodificar flujo Nikon: {used}")
            if out is not None:
                out[:] = decoded
                return out, int(used)
            return decoded, int(used)

        if out is None:
            out = np.empty((rows, width), dtype=np.uint16)
        seeds = [list(v) for v in self.header.seed_parities]
        r = MSBBitReader(stream)
        for y in range(rows):
            horizontal = [0, 0]
            for x in range(width):
                bits = self.huff.symbol(r)
                if bits == 0:
                    d = 0
                else:
                    val_b = r.read(bits)
                    d = val_b if (val_b & (1 << (bits - 1))) else val_b - ((1 << bits) - 1)
                parity = x & 1
                if x < 2:
                    val = seeds[y & 1][parity] + d
                    seeds[y & 1][parity] = val
                else:
                    val = horizontal[parity] + d
                if val < 0 or val >= 1 << 14:
                    raise CodecError(f"Predictor fuera de rango en (y={y}, x={x}), valor={val}")
                horizontal[parity] = val
                out[y, x] = val
        return out, r.consumed_bits
