"""Módulo de bajo nivel para contenedores TIFF/EP, Nikon NEF, MakerNote Type 3 y parches binarios.

Forma parte de la librería `nefinject`, proporcionando el soporte fundamental para el
análisis, manipulación e inyección transaccional en archivos RAW de cámaras Nikon (formato NEF).

Responsabilidades y componentes del módulo:
1. Análisis de Contenedores TIFF/EP (`TiffReader`, `TiffDirectory`, `TiffTag`):
   - Decodificación exhaustiva del encabezado TIFF (orden de bytes little/big endian),
     directorios IFD primarios, SubIFDs y directorios EXIF.
   - Resolución de valores inmediatos (<= 4 bytes) y desreferenciación de punteros a datos.
   - Detección de punteros a MakerNote y metadatos del sensor.

2. Inspección de MakerNote Nikon Type 3 (`MakerNoteIFDReader`, `MakerNoteTag`, `NikonMakerNoteInspector`):
   - Análisis estructurado del bloque privado de metadatos del fabricante Nikon ("Nikon\\x00\\x02...").
   - Decodificación de etiquetas críticas: balance de blancos (0x000c), recorte de sensor,
     configuración de compresión, curvas de tonos y datos de captura del fabricante.

3. Decodificación y Geometría del Mosaico CFA (`CfaGeometry`, `RawFrame`, `RawStatistics`):
   - Mapeo de dimensiones activas, márgenes ópticos y orientación del mosaico Bayer (RGGB).
   - Integración nativa opcional con LibRaw vía CTypes (`LibRawNativeProbe`) y `RawPyBackend`
     para validación de referencia del decodificador.

4. Planificación y Parcheo Transaccional (`PatchField`, `InjectionPlan`, `NefContainerPatcher`):
   - Generación de planes de inyección binaria con validación previa de integridad SHA-256.
   - Inyección atómica del flujo comprimido de píxeles Bayer sin modificar la estructura
     de directorios ni romper las firmas internas o compensaciones de punteros (StripOffsets
     y StripByteCounts).
   - Prevención de corrupción mediante verificaciones de redundancia antes y después del parcheo.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import io
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np
from PIL import Image

from .models import (
    CodecError,
    InjectionError,
    MakerNoteParseError,
    PayloadDecodeError,
    SCHEMA_VERSION,
    TiffParseError,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

MAX_IFDS = 128
MAX_TAGS = 4096
MAX_DEPTH = 8
MAX_VALUE_DECODE_BYTES = 4096
MAX_MAKER_BYTES = 32 * 1024 * 1024
MAX_MAKER_IFD_TAGS = 1024
MAX_MAKER_IFD_DEPTH = 2
MAX_MAKER_DECODE_BYTES = 1024
MAX_UINT32 = 0xFFFFFFFF
CHUNK_SIZE = 4 * 1024 * 1024

TAG_NAMES: dict[int, str] = {
    0x00FE: "NewSubfileType", 0x00FF: "SubfileType",
    0x0100: "ImageWidth", 0x0101: "ImageLength", 0x0102: "BitsPerSample",
    0x0103: "Compression", 0x0106: "PhotometricInterpretation",
    0x010E: "ImageDescription", 0x010F: "Make", 0x0110: "Model",
    0x0111: "StripOffsets", 0x0112: "Orientation", 0x0115: "SamplesPerPixel",
    0x0116: "RowsPerStrip", 0x0117: "StripByteCounts", 0x011A: "XResolution",
    0x011B: "YResolution", 0x011C: "PlanarConfiguration", 0x0128: "ResolutionUnit",
    0x0131: "Software", 0x0132: "DateTime", 0x013B: "Artist",
    0x0142: "TileWidth", 0x0143: "TileLength", 0x0144: "TileOffsets",
    0x0145: "TileByteCounts", 0x014A: "SubIFDs", 0x0201: "JPEGInterchangeFormat",
    0x0202: "JPEGInterchangeFormatLength", 0x0213: "YCbCrPositioning",
    0x0214: "ReferenceBlackWhite", 0x02BC: "XMP", 0x8298: "Copyright",
    0x829A: "ExposureTime", 0x829D: "FNumber", 0x8769: "ExifIFDPointer",
    0x8773: "ICCProfile", 0x8822: "ExposureProgram", 0x8825: "GPSIFDPointer",
    0x8827: "ISOSpeedRatings", 0x8830: "SensitivityType",
    0x9003: "DateTimeOriginal", 0x9004: "DateTimeDigitized",
    0x9201: "ShutterSpeedValue", 0x9202: "ApertureValue",
    0x9204: "ExposureBiasValue", 0x9205: "MaxApertureValue",
    0x9207: "MeteringMode", 0x9208: "LightSource", 0x9209: "Flash",
    0x920A: "FocalLength", 0x9217: "SensingMethod",
    0x927C: "MakerNote", 0x9286: "UserComment",
    0xA001: "ColorSpace", 0xA002: "PixelXDimension", 0xA003: "PixelYDimension",
    0xA005: "InteroperabilityIFDPointer", 0xA20E: "FocalPlaneXResolution",
    0xA20F: "FocalPlaneYResolution", 0xA210: "FocalPlaneResolutionUnit",
    0xA302: "CFAPatternExif", 0xA403: "WhiteBalance", 0xA405: "FocalLengthIn35mmFilm",
    0x828D: "CFARepeatPatternDim", 0x828E: "CFAPattern",
    0xC612: "DNGVersion", 0xC614: "UniqueCameraModel",
    0xC619: "BlackLevelRepeatDim", 0xC61A: "BlackLevel", 0xC61D: "WhiteLevel",
    0xC61F: "DefaultCropOrigin", 0xC620: "DefaultCropSize",
    0xC621: "ColorMatrix1", 0xC622: "ColorMatrix2",
    0xC623: "CameraCalibration1", 0xC624: "CameraCalibration2",
    0xC627: "AnalogBalance", 0xC628: "AsShotNeutral",
    0xC62A: "BaselineExposure", 0xC65A: "CalibrationIlluminant1",
    0xC65B: "CalibrationIlluminant2", 0xC68B: "OriginalRawFileName",
    0xC68D: "ActiveArea", 0xC6F5: "ExtraCameraProfiles",
    0xC6F8: "ProfileName", 0xC6F9: "ProfileHueSatMapDims",
    0xC6FA: "ProfileHueSatMapData1", 0xC6FB: "ProfileHueSatMapData2",
    0xC6FC: "ProfileToneCurve", 0xC6FD: "ProfileEmbedPolicy",
    0xC714: "ForwardMatrix1", 0xC715: "ForwardMatrix2",
    0xC725: "ProfileLookTableDims", 0xC726: "ProfileLookTableData",
}

NAME_TO_TAG: dict[str, int] = {v: k for k, v in TAG_NAMES.items()}

TYPE_INFO: dict[int, tuple[str, int, str | None]] = {
    1: ("BYTE", 1, "B"), 2: ("ASCII", 1, None),
    3: ("SHORT", 2, "H"), 4: ("LONG", 4, "I"),
    5: ("RATIONAL", 8, "II"), 6: ("SBYTE", 1, "b"),
    7: ("UNDEFINED", 1, None), 8: ("SSHORT", 2, "h"),
    9: ("SLONG", 4, "i"), 10: ("SRATIONAL", 8, "ii"),
    11: ("FLOAT", 4, "f"), 12: ("DOUBLE", 8, "d"),
    13: ("IFD", 4, "I"),
}

COMPRESSION: dict[int, str] = {
    1: "none", 5: "lzw", 6: "old_style_jpeg", 7: "jpeg",
    8: "deflate", 32769: "packed_or_vendor_specific",
    34713: "nikon_proprietary_compression",
}
PHOTOMETRIC: dict[int, str] = {2: "RGB", 6: "YCbCr", 32803: "CFA"}
CFA_COLORS: dict[int, str] = {0: "R", 1: "G", 2: "B", 3: "C", 4: "M", 5: "Y", 6: "W"}
SENSITIVE_TAG_IDS: set[int] = {0x013B, 0x8298, 0x9286, 0x02BC}

NIKON_NAMES: dict[int, str] = {
    0x0001: "MakerNoteVersion", 0x0002: "ISO", 0x0004: "ColorMode",
    0x0005: "Quality", 0x0007: "WhiteBalanceMode", 0x0008: "FocusMode",
    0x0009: "FlashSetting", 0x000B: "WhiteBalanceFineTune", 0x000C: "WB_RBLevels",
    0x000D: "ProgramShift", 0x000E: "ExposureDifference", 0x0011: "NikonPreviewIFD",
    0x0012: "FlashExposureComp", 0x0013: "ISOSetting", 0x0017: "ExposureBracketComp",
    0x0018: "FlashBracketComp", 0x0019: "ExposureBracketValue",
    0x001B: "CropHiSpeed", 0x001C: "ExposureTuning",
    0x001D: "SerialNumber", 0x001E: "ColorSpace", 0x001F: "VRInfo",
    0x0022: "ActiveDLighting", 0x0023: "PictureControl", 0x0024: "WorldTime",
    0x0025: "ISOInfo", 0x002A: "VignetteControl", 0x002B: "DistortInfo",
    0x002C: "Unknown_0x002C", 0x0032: "Unknown_0x0032",
    0x0083: "LensType", 0x0084: "Lens", 0x0087: "FlashMode",
    0x0089: "ShootingMode", 0x008A: "AutoBracketRelease", 0x008B: "LensFStops",
    0x008C: "ContrastCurve", 0x0091: "ShotInfo", 0x0093: "NEFCompression",
    0x0095: "NoiseReduction", 0x0096: "NEFLinearizationTable",
    0x0097: "ColorBalance", 0x0098: "LensData", 0x0099: "RawImageCenter",
    0x009A: "SensorPixelSize", 0x009E: "RetouchHistory", 0x00A0: "SerialNumber2",
    0x00A2: "ImageDataSize", 0x00A4: "Unknown_0x00A4", 0x00A7: "ShutterCount",
    0x00A8: "FlashInfo", 0x00B1: "HighISONoiseReduction", 0x00B6: "Unknown_0x00B6",
    0x00B7: "AFInfo2", 0x00B8: "FileInfo", 0x00B9: "AFTune",
    0x00BA: "Unknown_0x00BA", 0x00BB: "Unknown_0x00BB",
    0x00BC: "Unknown_0x00BC", 0x00BF: "Unknown_0x00BF", 0x00C0: "Unknown_0x00C0",
}
PREVIEW_NAMES: dict[int, str] = {
    0x0103: "Compression", 0x011A: "XResolution", 0x011B: "YResolution",
    0x0128: "ResolutionUnit", 0x0201: "JPEGInterchangeFormat",
    0x0202: "JPEGInterchangeFormatLength", 0x0213: "YCbCrPositioning",
}
NEF_COMPRESSION: dict[int, str] = {
    1: "lossy_type_1", 2: "uncompressed", 3: "lossless", 4: "lossy_type_2",
    5: "striped_packed_12", 6: "uncompressed_reduced_12",
    7: "unpacked_12", 8: "small", 9: "packed_12", 10: "packed_14",
}
PRIVATE_FIELDS: set[int] = {0x001D, 0x00A0}
OPAQUE_FIELDS: set[int] = {
    0x0023, 0x002C, 0x008C, 0x0091, 0x0096,
    0x0097, 0x0098, 0x00A8, 0x00B7, 0x00B8,
    0x00BA, 0x00BC,
}
VERSIONED_FIELDS: set[int] = {
    0x0001, 0x001F, 0x0023, 0x002B, 0x002C,
    0x0091, 0x0097, 0x0098, 0x00A8, 0x00B7,
    0x00B8, 0x00BA, 0x00BC,
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _scalar(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rational_decimal(v: Any) -> float | int | None:
    if isinstance(v, dict):
        return v.get("decimal")
    if isinstance(v, (int, float)):
        return v
    return None


def _ascii_version(data: bytes) -> str | None:
    first = data[:4]
    if len(first) == 4 and all(0x20 <= x < 0x7F for x in first):
        return first.decode("ascii")
    return None


# ---------------------------------------------------------------------------
# Lector TIFF Clásico (NEF y DNG)
# ---------------------------------------------------------------------------

@dataclass
class TiffTag:
    code: int
    dtype: int
    count: int
    entry_offset: int
    value_field_offset: int
    data_offset: int
    byte_size: int
    inline: bool
    data: bytes | None = field(default=None, repr=False)
    validation_error: str | None = None

    @property
    def name(self) -> str:
        return TAG_NAMES.get(self.code, f"Unknown_0x{self.code:04X}")

    @property
    def type_name(self) -> str:
        return TYPE_INFO.get(self.dtype, ("UNKNOWN", 0, None))[0]

    def decode(self, endian: str, max_bytes: int = MAX_VALUE_DECODE_BYTES) -> Any:
        if self.data is None or self.validation_error or self.byte_size > max_bytes:
            return None
        if self.dtype == 2:
            return self.data.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        if self.dtype == 7:
            return None
        if self.dtype not in TYPE_INFO:
            return None
        _, _, fmt = TYPE_INFO[self.dtype]
        if fmt is None:
            return list(self.data) if self.count <= 32 else None
        try:
            if self.dtype in (5, 10):
                vals = []
                for i in range(self.count):
                    n, d = struct.unpack_from(endian + fmt, self.data, i * 8)
                    vals.append({"numerator": n, "denominator": d, "decimal": _safe_ratio(n, d)})
                return _scalar(vals)
            vals = list(struct.unpack(endian + fmt * self.count, self.data))
            return _scalar(vals)
        except (struct.error, OverflowError, ValueError):
            return None

    def json_dict(
        self,
        endian: str,
        *,
        include_sensitive: bool = False,
        redact_value: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": f"0x{self.code:04X}",
            "name": self.name,
            "type_id": self.dtype,
            "type": self.type_name,
            "count": self.count,
            "ifd_entry_offset": self.entry_offset,
            "value_field_offset": self.value_field_offset,
            "value_data_offset": self.data_offset,
            "value_byte_count": self.byte_size,
            "value_is_inline": self.inline,
        }
        if self.validation_error:
            result["validation_error"] = self.validation_error
        elif redact_value:
            result["value_summary"] = "contenido GPS omitido"
        elif self.code == 0x927C:
            result["value_summary"] = "Nikon MakerNote / bloque opaco no interpretado"
        elif self.code in SENSITIVE_TAG_IDS and not include_sensitive:
            result["value_summary"] = "contenido omitido por defecto"
        elif self.byte_size > MAX_VALUE_DECODE_BYTES:
            result["value_summary"] = "contenido grande no extraído"
        else:
            value = self.decode(endian)
            if value is not None:
                result["value"] = value
        return result


@dataclass
class TiffDirectory:
    path: str
    role: str
    offset: int
    tags: dict[int, TiffTag]
    next_ifd_offset: int
    parent_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def tag(self, code: int) -> TiffTag | None:
        return self.tags.get(code)

    def tag_by_name(self, name: str) -> TiffTag | None:
        code = NAME_TO_TAG.get(name)
        return self.tags.get(code) if code is not None else None

    def value(self, code: int, endian: str, default: Any = None) -> Any:
        tag = self.tags.get(code)
        return tag.decode(endian) if tag else default

    def num(self, code: int, endian: str) -> int | None:
        val = self.value(code, endian)
        return int(val) if isinstance(val, (int, float)) else None


class TiffReader:
    """Lector binario de TIFF clásico (endianness II/MM) para NEF y DNG."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.endian = ""
        self.first_ifd_offset = 0
        self.directories: list[TiffDirectory] = []
        self.warnings: list[str] = []
        self._visited: set[int] = set()
        self._handle: BinaryIO | None = None

    def _read(self, offset: int, count: int) -> bytes:
        if offset < 0 or count < 0 or offset + count > self.size:
            raise TiffParseError(f"Rango TIFF fuera del archivo: [{offset}, {offset + count}) / {self.size}")
        assert self._handle is not None
        self._handle.seek(offset)
        data = self._handle.read(count)
        if len(data) != count:
            raise TiffParseError(f"Lectura incompleta en {offset}: {len(data)} de {count}")
        return data

    def read_tag_bytes(self, tag: TiffTag) -> bytes:
        """Lee los bytes completos de un TiffTag bajo demanda (incluso si > 4096 bytes)."""
        if tag.data is not None and len(tag.data) == tag.byte_size:
            return tag.data
        if tag.validation_error or tag.data_offset < 0 or tag.data_offset + tag.byte_size > self.size:
            raise TiffParseError(f"Tag 0x{tag.code:04X} fuera de rango")
        with self.path.open("rb") as f:
            f.seek(tag.data_offset)
            data = f.read(tag.byte_size)
        if len(data) != tag.byte_size:
            raise TiffParseError(f"Tag 0x{tag.code:04X} truncado")
        return data

    def _read_ifd(
        self, offset: int, path: str, role: str, depth: int, parent: str | None
    ) -> TiffDirectory | None:
        if offset == 0:
            return None
        if depth > MAX_DEPTH:
            self.warnings.append(f"Profundidad máxima de IFD excedida en {path}")
            return None
        if offset in self._visited:
            self.warnings.append(f"IFD compartido o referencia cíclica en {path}: offset {offset}")
            return None
        if len(self.directories) >= MAX_IFDS:
            raise TiffParseError("Demasiados IFD; posible archivo corrupto")
        self._visited.add(offset)
        num = struct.unpack(self.endian + "H", self._read(offset, 2))[0]
        if num > MAX_TAGS:
            raise TiffParseError(f"IFD {path} con {num} entradas (> {MAX_TAGS})")
        table = self._read(offset + 2, 12 * num + 4)
        tags: dict[int, TiffTag] = {}
        local_warnings: list[str] = []
        for i in range(num):
            entry_offset = offset + 2 + i * 12
            code, typ, count, raw_ptr = struct.unpack_from(self.endian + "HHII", table, 12 * i)
            value_field_offset = entry_offset + 8
            if typ not in TYPE_INFO:
                local_warnings.append(f"Tag 0x{code:04X}: tipo TIFF desconocido {typ}")
                tags[code] = TiffTag(
                    code, typ, count, entry_offset, value_field_offset,
                    raw_ptr, 0, False, None, "tipo TIFF desconocido",
                )
                continue
            byte_size = TYPE_INFO[typ][1] * count
            inline = byte_size <= 4
            data_offset = value_field_offset if inline else raw_ptr
            data = None
            error = None
            if data_offset < 0 or data_offset + byte_size > self.size:
                error = "valor fuera del archivo"
                local_warnings.append(f"Tag 0x{code:04X}: {error}")
            elif byte_size <= MAX_VALUE_DECODE_BYTES and code != 0x927C:
                data = self._read(data_offset, byte_size)
            tags[code] = TiffTag(
                code, typ, count, entry_offset, value_field_offset,
                data_offset, byte_size, inline, data, error,
            )
        next_ifd = struct.unpack_from(self.endian + "I", table, 12 * num)[0]
        current = TiffDirectory(path, role, offset, tags, next_ifd, parent, local_warnings)
        self.directories.append(current)
        for code, child_role in (
            (0x014A, "subifd"),
            (0x8769, "exif"),
            (0x8825, "gps"),
            (0xA005, "interop"),
        ):
            tag = current.tag(code)
            if tag is None:
                continue
            offs = _as_list(tag.decode(self.endian))
            for i, off in enumerate(offs):
                if not isinstance(off, int) or off <= 0:
                    local_warnings.append(f"Puntero IFD inválido para {tag.name}: {off}")
                    continue
                suffix = f"SubIFD{i}" if code == 0x014A else child_role.upper()
                self._read_ifd(off, path + "/" + suffix, child_role, depth + 1, path)
        return current

    def parse(self) -> "TiffReader":
        self.directories.clear()
        self._visited.clear()
        self.warnings.clear()
        with self.path.open("rb") as f:
            self._handle = f
            header = self._read(0, 8)
            if header[:2] == b"MM":
                self.endian = ">"
            elif header[:2] == b"II":
                self.endian = "<"
            else:
                raise TiffParseError("Cabecera TIFF no válida: faltan II/MM")
            magic, self.first_ifd_offset = struct.unpack(self.endian + "HI", header[2:8])
            if magic == 43:
                raise TiffParseError("BigTIFF no implementado; requiere entradas de 20 bytes")
            if magic != 42:
                raise TiffParseError(f"Identificador TIFF desconocido: {magic}")
            off, idx = self.first_ifd_offset, 0
            while off:
                d = self._read_ifd(off, f"IFD{idx}", "primary", 0, None)
                if d is None:
                    break
                off = d.next_ifd_offset
                idx += 1
            self._handle = None
        return self

    def get(self, path: str) -> TiffDirectory | None:
        return next((d for d in self.directories if d.path == path), None)


def _segment(offset: int, length: int, file_size: int, kind: str, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "offset": offset,
        "byte_count": length,
        "end_offset_exclusive": offset + length,
        "inside_file": bool(offset >= 0 and length > 0 and offset + length <= file_size),
    }


def _ranges(directory: TiffDirectory, reader: TiffReader) -> list[dict[str, Any]]:
    e = reader.endian
    segments: list[dict[str, Any]] = []
    for off_tag, size_tag, kind in (
        (0x0111, 0x0117, "strip"),
        (0x0144, 0x0145, "tile"),
        (0x0201, 0x0202, "jpeg_preview"),
    ):
        offsets = _as_list(directory.value(off_tag, e))
        lengths = _as_list(directory.value(size_tag, e))
        if not offsets and not lengths:
            continue
        if len(offsets) != len(lengths):
            reader.warnings.append(f"{directory.path}: {kind}, número de offsets y tamaños diferente")
        for i, (off, size) in enumerate(zip(offsets, lengths)):
            if isinstance(off, int) and isinstance(size, int):
                segments.append(_segment(off, size, reader.size, kind, i))
    return segments


def _jpeg_sof_dimensions(path: Path, offset: int, length: int) -> tuple[int, int] | None:
    if offset < 0 or length < 4:
        return None
    with path.open("rb") as f:
        f.seek(offset)
        b = f.read(min(length, 131072))
    if len(b) < 4 or b[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(b):
        if b[i] != 0xFF:
            return None
        while i < len(b) and b[i] == 0xFF:
            i += 1
        if i >= len(b):
            break
        marker = b[i]
        i += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0xD9, 0xDA):
            break
        if i + 2 > len(b):
            break
        marker_length = int.from_bytes(b[i : i + 2], "big")
        if marker_length < 2 or i + marker_length > len(b):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if marker_length < 7:
                return None
            height = int.from_bytes(b[i + 3 : i + 5], "big")
            width = int.from_bytes(b[i + 5 : i + 7], "big")
            return (width, height) if width and height else None
        i += marker_length
    return None


def _cfa(directory: TiffDirectory, endian: str) -> dict[str, Any] | None:
    dims = _as_list(directory.value(0x828D, endian))
    tag = directory.tag(0x828E)
    if len(dims) != 2 or tag is None or tag.data is None:
        return None
    rows, cols = dims
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
        return None
    if rows * cols != len(tag.data):
        return {"repeat_pattern_dim": dims, "error": "dimensiones CFA y muestras no coinciden"}
    codes = list(tag.data)
    names = [CFA_COLORS.get(c, f"{c}") for c in codes]
    return {
        "repeat_pattern_dim": [rows, cols],
        "color_codes": [codes[r * cols : (r + 1) * cols] for r in range(rows)],
        "color_names": [names[r * cols : (r + 1) * cols] for r in range(rows)],
        "pattern": "".join(names) if rows * cols <= 16 else None,
    }


class NefInspector:
    """Inspección estructural no destructiva de archivos Nikon NEF."""

    def __init__(
        self,
        path: str | Path,
        *,
        include_sensitive: bool = False,
        compute_sha256: bool = False,
        precomputed_sha256: str | None = None,
        reader: TiffReader | None = None,
    ) -> None:
        self.path = Path(path)
        self.include_sensitive = include_sensitive
        self.compute_sha256 = compute_sha256
        self.precomputed_sha256 = precomputed_sha256
        self.reader: TiffReader | None = reader

    def _image_info(self, d: TiffDirectory) -> dict[str, Any]:
        assert self.reader
        reader = self.reader
        e = reader.endian
        w, h = d.num(0x0100, e), d.num(0x0101, e)
        comp, photo = d.num(0x0103, e), d.num(0x0106, e)
        bits = _as_list(d.value(0x0102, e))
        subfile = d.num(0x00FE, e)
        cfa = _cfa(d, e)
        segments = _ranges(d, reader)
        dimensions_source = "ifd_tags" if w and h else "unknown"
        if not (w and h):
            for segment in segments:
                if segment["kind"] == "jpeg_preview" and segment["inside_file"]:
                    jpeg_dims = _jpeg_sof_dimensions(self.path, segment["offset"], segment["byte_count"])
                    if jpeg_dims:
                        w, h = jpeg_dims
                        dimensions_source = "embedded_jpeg_sof"
                        break
        looks_raw = photo == 32803 or (cfa is not None and subfile in (None, 0))
        is_preview = (
            (subfile is not None and subfile & 1 == 1)
            or photo in (2, 6)
            or any(seg["kind"] == "jpeg_preview" for seg in segments)
        )
        kind = "raw_candidate" if looks_raw else "preview" if is_preview else "unclassified"
        return {
            "ifd_path": d.path,
            "ifd_offset": d.offset,
            "kind": kind,
            "width": w,
            "height": h,
            "dimensions_source": dimensions_source,
            "pixel_count": w * h if w and h else None,
            "new_subfile_type": subfile,
            "samples_per_pixel": d.num(0x0115, e),
            "bits_per_sample": bits,
            "compression": {"code": comp, "name": COMPRESSION.get(comp, "unknown")},
            "photometric": {"code": photo, "name": PHOTOMETRIC.get(photo, "unknown")},
            "rows_per_strip": d.num(0x0116, e),
            "cfa": cfa,
            "segments": segments,
            "payload_total_bytes": sum(s["byte_count"] for s in segments),
        }

    def _capture_info(self) -> dict[str, Any]:
        assert self.reader
        reader = self.reader
        root = reader.get("IFD0")
        if root is None:
            return {}
        exif = reader.get("IFD0/EXIF")

        def val(tag: int) -> Any:
            return exif.value(tag, reader.endian) if exif else None

        return {
            "make": root.value(0x010F, reader.endian),
            "model": root.value(0x0110, reader.endian),
            "firmware_software": root.value(0x0131, reader.endian),
            "exposure_time_s": _rational_decimal(val(0x829A)),
            "aperture_f_number": _rational_decimal(val(0x829D)),
            "iso": val(0x8827),
            "focal_length_mm": _rational_decimal(val(0x920A)),
            "white_balance_exif_code": val(0xA403),
            "date_time_original": val(0x9003) if self.include_sensitive else None,
            "maker_note": self._maker_note_info(exif),
        }

    @staticmethod
    def _maker_note_info(exif: TiffDirectory | None) -> dict[str, Any] | None:
        if exif is None:
            return None
        tag = exif.tag(0x927C)
        if not tag:
            return None
        return {
            "present": True,
            "byte_count": tag.byte_size,
            "offset": tag.data_offset,
            "decoded": False,
            "note": "MakerNote propietario: conservar íntegro al crear un futuro inyector.",
        }

    @staticmethod
    def _feasibility(raw: dict[str, Any] | None) -> dict[str, Any]:
        if raw is None:
            return {
                "status": "blocked_no_raw_candidate",
                "direct_payload_patch": False,
                "reason_codes": ["RAW_IFD_NOT_FOUND"],
            }
        compression = raw["compression"]["code"]
        width, height = raw["width"], raw["height"]
        bits = raw["bits_per_sample"]
        n_pix = width * height if width and height else None
        bit_depth = bits[0] if len(bits) == 1 and isinstance(bits[0], int) else None
        packed = (n_pix * bit_depth + 7) // 8 if n_pix and bit_depth else None
        container16 = n_pix * 2 if n_pix else None
        total = raw["payload_total_bytes"]
        segments = raw["segments"]
        report: dict[str, Any] = {
            "status": None,
            "direct_payload_patch": False,
            "reason_codes": [],
            "read_only_inspection": True,
            "compression_mode_lossless_or_lossy": "not_determined",
            "sample_count": n_pix,
            "expected_uncompressed_packed_bytes_no_padding": packed,
            "expected_uncompressed_uint16_bytes": container16,
            "payload_total_bytes": total,
            "payload_vs_tightly_packed_ratio": round(total / packed, 6) if packed else None,
            "single_segment": len(segments) == 1,
            "segments_within_file": bool(segments) and all(x["inside_file"] for x in segments),
        }
        if compression == 34713:
            report["status"] = "blocked_nikon_proprietary_compression"
            report["reason_codes"] = ["NIKON_COMPRESSION_34713", "ENCODER_REQUIRED"]
            report["next_step"] = "Solicitar un NEF sin compresión de este modelo o desarrollar un códec compatible."
        elif compression == 32769:
            report["status"] = "blocked_packing_unknown"
            report["reason_codes"] = ["PACKED_VENDOR_FORMAT", "PACKING_VALIDATION_REQUIRED"]
            report["next_step"] = "Identificar empaquetamiento y rellenos antes de modificar datos."
        elif compression == 1:
            report["status"] = "conditional_uncompressed_layout_verification_required"
            report["reason_codes"] = ["PACKING_VALIDATION_REQUIRED", "ACTIVE_AREA_VALIDATION_REQUIRED"]
            if not report["segments_within_file"]:
                report["status"] = "blocked_invalid_payload_ranges"
                report["reason_codes"] = ["INVALID_SEGMENT_BOUNDS"]
            report["next_step"] = "Verificar endianness, orden de muestras, empaquetamiento y márgenes."
        else:
            report["status"] = "blocked_unsupported_compression"
            report["reason_codes"] = ["UNSUPPORTED_COMPRESSION"]
            report["next_step"] = "Obtener una muestra RAW sin comprimir."
        return report

    def inspect(self) -> dict[str, Any]:
        if self.reader is None:
            self.reader = TiffReader(self.path).parse()
        r = self.reader
        capture = self._capture_info()
        images = [self._image_info(d) for d in r.directories if d.role in ("primary", "subifd")]
        raw_candidates = [d for d in images if d["kind"] == "raw_candidate"]
        raw_candidates.sort(key=lambda d: ((d["pixel_count"] or 0), d["payload_total_bytes"]), reverse=True)
        raw = raw_candidates[0] if raw_candidates else None
        previews = [d for d in images if d["kind"] == "preview"]
        warnings = list(r.warnings)
        for d in r.directories:
            warnings.extend(f"{d.path}: {s}" for s in d.warnings)
        if raw is None:
            warnings.append("No se encontró un IFD CFA clasificable como RAW.")
        elif raw["compression"]["code"] == 34713:
            warnings.append(
                "Compresión Nikon 34713: no se puede inferir solo de este código si es reversible o con pérdidas."
            )
        if str(capture.get("make") or "").upper().find("NIKON") == -1:
            warnings.append("Fabricante no identificado como Nikon: comprobar que sea un NEF.")
        if raw and not all(s["inside_file"] for s in raw["segments"]):
            warnings.append("Uno o más segmentos RAW quedan fuera de los límites del fichero.")
        sha = (
            self.precomputed_sha256
            if self.precomputed_sha256 is not None
            else (sha256_file(self.path) if self.compute_sha256 else None)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": {
                "name": "nef_inspector",
                "version": "0.1.0",
                "read_only": True,
                "raw_pixels_decoded": False,
                "maker_notes_decoded": False,
            },
            "source": {
                "filename": self.path.name,
                "path": str(self.path.absolute()),
                "byte_count": r.size,
                "sha256": sha,
            },
            "format": {
                "family": "TIFF/EP - Nikon NEF" if "NIKON" in str(capture.get("make") or "").upper() else "TIFF",
                "byte_order": "big_endian_MM" if r.endian == ">" else "little_endian_II",
                "tiff_variant": "classic_tiff_32bit_offsets",
                "first_ifd_offset": r.first_ifd_offset,
            },
            "camera": capture,
            "raw": raw,
            "raw_candidate_count": len(raw_candidates),
            "previews": previews,
            "injection_assessment": self._feasibility(raw),
            "tiff": {
                "ifd_count": len(r.directories),
                "ifds": [
                    {
                        "path": d.path,
                        "role": d.role,
                        "ifd_offset": d.offset,
                        "parent_path": d.parent_path,
                        "next_ifd_offset": d.next_ifd_offset,
                        "tag_count": len(d.tags),
                        "tags": [
                            d.tags[k].json_dict(
                                r.endian,
                                include_sensitive=self.include_sensitive,
                                redact_value=(d.role == "gps"),
                            )
                            for k in sorted(d.tags)
                        ],
                    }
                    for d in r.directories
                ],
            },
            "warnings": warnings,
            "interpretation_notes": [
                "Los offsets son posiciones absolutas de bytes desde el inicio del fichero.",
                "value_field_offset es el campo de 4 bytes de cada entrada IFD; value_data_offset puede apuntar a otro lugar.",
                "StripOffsets/TileOffsets son valores de tags; no confundir con value_data_offset.",
                "Los niveles de negro, saturación, WB y matrices que no se lean de tags estándar permanecen desconocidos.",
                "La presencia del identificador Nikon no garantiza compatibilidad con reveladores comerciales.",
            ],
        }

    def write_json(self, output: str | Path, *, overwrite: bool = True) -> dict[str, Any]:
        result = self.inspect()
        write_json_atomic(output, result, overwrite=overwrite)
        return result


# ---------------------------------------------------------------------------
# Lector Nikon MakerNote Type 3
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MakerNoteTag:
    id: int
    type_id: int
    count: int
    entry_offset: int
    value_field_offset: int
    data_offset: int
    value_byte_count: int
    inline: bool
    offset_valid: bool
    value: bytes | None = None
    error: str | None = None

    def decode(self, endian: str) -> Any:
        if self.error or self.value is None or self.value_byte_count > MAX_MAKER_DECODE_BYTES:
            return None
        if self.type_id == 2:
            return self.value.partition(b"\0")[0].decode("utf-8", "replace")
        if self.type_id == 7:
            return None
        info = TYPE_INFO.get(self.type_id)
        if not info:
            return None
        fmt = info[2]
        if fmt is None:
            return list(self.value) if self.count <= 16 else None
        try:
            if self.type_id in (5, 10):
                vals = [
                    {
                        "numerator": n,
                        "denominator": d,
                        "decimal": (n / d if d else None),
                    }
                    for i in range(self.count)
                    for n, d in [struct.unpack_from(endian + fmt, self.value, i * 8)]
                ]
            else:
                vals = list(struct.unpack(endian + fmt * self.count, self.value))
            return vals[0] if len(vals) == 1 else vals
        except (struct.error, ValueError, OverflowError):
            return None

    def to_json(
        self,
        endian: str,
        *,
        names: dict[int, str] | None = None,
        show_sensitive: bool = False,
    ) -> dict[str, Any]:
        name = (names or NIKON_NAMES).get(self.id, f"Unknown_0x{self.id:04X}")
        result: dict[str, Any] = {
            "id": f"0x{self.id:04X}",
            "name": name,
            "type_id": self.type_id,
            "type": TYPE_INFO.get(self.type_id, ("UNKNOWN",))[0],
            "count": self.count,
            "ifd_entry_offset": self.entry_offset,
            "value_field_offset": self.value_field_offset,
            "value_data_offset": self.data_offset,
            "value_byte_count": self.value_byte_count,
            "value_is_inline": self.inline,
            "range_valid": self.offset_valid,
        }
        if self.error:
            result["error"] = self.error
            return result
        if self.id in PRIVATE_FIELDS and not show_sensitive:
            result["value_summary"] = "redacted_sensitive"
            return result
        if self.id in OPAQUE_FIELDS:
            result["value_summary"] = "opaque_not_decoded"
            if self.value is not None and self.id in VERSIONED_FIELDS:
                result["version_prefix"] = _ascii_version(self.value)
            return result
        if self.type_id == 7 and self.id in VERSIONED_FIELDS:
            result["value_summary"] = "undefined_binary"
            if self.value is not None:
                result["version_prefix"] = _ascii_version(self.value)
            return result
        val = self.decode(endian)
        if val is None:
            result["value_summary"] = "not_decoded"
        else:
            result["value"] = val
        return result


class MakerNoteIFDReader:
    """Lector TIFF embebido Nikon Type 3 con validación de rangos físicos."""

    def __init__(self, data: bytes, file_offset: int) -> None:
        if len(data) > MAX_MAKER_BYTES:
            raise MakerNoteParseError(f"MakerNote demasiado grande: {len(data)} bytes")
        self.data = data
        self.file_offset = file_offset
        self.region_start = file_offset
        self.region_end = file_offset + len(data)
        self.version: str | None = None
        self.kind: str = ""
        self.tiff_relative_base: int | None = None
        self.endian: str = ""
        self.first_ifd_relative: int | None = None
        self.warnings: list[str] = []
        self._prepare()

    def _prepare(self) -> None:
        d = self.data
        if not d.startswith(b"Nikon\x00"):
            raise MakerNoteParseError("No se detecta firma Nikon en MakerNote")
        if len(d) < 8:
            raise MakerNoteParseError("MakerNote Nikon truncado")
        if d[6] == 2:
            self.kind = "nikon_type3_embedded_tiff"
            if len(d) < 18:
                raise MakerNoteParseError("MakerNote Nikon Type3 truncado (<18 bytes)")
            self.tiff_relative_base = 10
            marker = d[10:12]
            if marker not in (b"MM", b"II"):
                raise MakerNoteParseError("No hay cabecera TIFF II/MM en MakerNote +10")
            self.endian = ">" if marker == b"MM" else "<"
            magic, first = struct.unpack_from(self.endian + "HI", d, 12)
            if magic != 42:
                raise MakerNoteParseError(f"Magic TIFF embebido inválido: {magic}")
            self.first_ifd_relative = first
        elif d[6] == 1:
            self.kind = "nikon_type2_not_implemented"
            self.version = None
            self.warnings.append("Nikon Type2 reconocido pero su tabla IFD no está implementada")
            return
        else:
            self.kind = "nikon_unknown"
            self.warnings.append(f"Versión Nikon desconocida: {d[6]}")
            return
        self.version = f"{d[6]}.{d[7]}"

    def _slice_rel(self, relative: int, length: int) -> bytes:
        if (
            self.tiff_relative_base is None
            or relative < 0
            or length < 0
            or self.tiff_relative_base + relative + length > len(self.data)
        ):
            raise MakerNoteParseError(f"Puntero fuera de MakerNote: +{relative}, tamaño {length}")
        start = self.tiff_relative_base + relative
        return self.data[start : start + length]

    def ifd(self, ifd_rel: int, *, depth: int = 0) -> dict[str, Any]:
        if depth > MAX_MAKER_IFD_DEPTH:
            raise MakerNoteParseError("Máxima profundidad IFD excedida")
        if not self.endian or self.tiff_relative_base is None:
            raise MakerNoteParseError("Tipo MakerNote no soportado")
        count_b = self._slice_rel(ifd_rel, 2)
        n = struct.unpack(self.endian + "H", count_b)[0]
        if n > MAX_MAKER_IFD_TAGS:
            raise MakerNoteParseError(f"Exceso de etiquetas en IFD: {n}")
        entries = self._slice_rel(ifd_rel + 2, n * 12 + 4)
        tags: dict[int, MakerNoteTag] = {}
        warnings: list[str] = []
        ifd_abs = self.file_offset + self.tiff_relative_base + ifd_rel
        for i in range(n):
            position = 12 * i
            tag_id, dtype, count, raw_ptr = struct.unpack_from(self.endian + "HHII", entries, position)
            entry_abs = ifd_abs + 2 + position
            val_field_abs = entry_abs + 8
            if tag_id in tags:
                warnings.append(f"Tag duplicado 0x{tag_id:04X}; se conserva el primero")
                continue
            if dtype not in TYPE_INFO:
                tags[tag_id] = MakerNoteTag(
                    tag_id, dtype, count, entry_abs, val_field_abs,
                    val_field_abs, 0, True, False, None, "unknown_type",
                )
                warnings.append(f"0x{tag_id:04X}: tipo TIFF desconocido {dtype}")
                continue
            unit = TYPE_INFO[dtype][1]
            length = count * unit
            inline = length <= 4
            data_rel = (ifd_rel + 2 + position + 8) if inline else raw_ptr
            data_abs = self.file_offset + self.tiff_relative_base + data_rel
            is_valid = data_rel >= 0 and self.tiff_relative_base + data_rel + length <= len(self.data)
            value = None
            error = None
            if not is_valid:
                error = "outside_makernote"
                warnings.append(f"0x{tag_id:04X}: offset {data_abs} (n={length}) fuera de MakerNote")
            elif length <= MAX_MAKER_DECODE_BYTES:
                value = self._slice_rel(data_rel, length)
            tags[tag_id] = MakerNoteTag(
                tag_id, dtype, count, entry_abs, val_field_abs,
                data_abs, length, inline, is_valid, value, error,
            )
        next_rel = struct.unpack_from(self.endian + "I", entries, n * 12)[0]
        return {
            "ifd_absolute_offset": ifd_abs,
            "ifd_relative_offset": ifd_rel,
            "tag_count": n,
            "tags": tags,
            "next_ifd_relative_offset": next_rel,
            "warnings": warnings,
        }


class NikonMakerNoteInspector:
    """Inspección Nikon MakerNote sin re-parsear ni re-hashear el NEF cuando se suministra contexto."""

    def __init__(
        self,
        path: str | Path,
        *,
        inspection: dict[str, Any] | None = None,
        show_sensitive: bool = False,
        compute_sha256: bool = False,
        precomputed_sha256: str | None = None,
        reader: TiffReader | None = None,
    ) -> None:
        self.path = Path(path)
        self.inspection = inspection
        self.show_sensitive = show_sensitive
        self.compute_sha256 = compute_sha256
        self.precomputed_sha256 = precomputed_sha256
        self.reader = reader
        self.last_makernote_bytes: bytes | None = None

    @staticmethod
    def _provenance(tag: MakerNoteTag | None) -> dict[str, Any] | None:
        if not tag:
            return None
        return {
            "tag_id": f"0x{tag.id:04X}",
            "data_offset": tag.data_offset,
            "ifd_entry_offset": tag.entry_offset,
            "type_id": tag.type_id,
        }

    @staticmethod
    def _validate_inspection(
        inspection: dict[str, Any] | None,
        path: Path,
        raw_tag: Any,
        make: str | None,
        model: str | None,
    ) -> list[str]:
        warnings: list[str] = []
        if inspection is None:
            return warnings
        source = inspection.get("source", {})
        if source.get("byte_count") != path.stat().st_size:
            warnings.append("INSPECTION_SIZE_MISMATCH: tamaño diferente; no reutilizar offsets JSON")
        if source.get("filename") != path.name:
            warnings.append("INSPECTION_FILENAME_MISMATCH: el JSON pertenece a otro NEF")
        cam = inspection.get("camera", {})
        if cam.get("model") not in (None, model) or cam.get("make") not in (None, make):
            warnings.append("INSPECTION_CAMERA_MISMATCH: fabricante o modelo distintos")
        stored = cam.get("maker_note") or {}
        if raw_tag is not None and (
            stored.get("offset") not in (None, raw_tag.data_offset)
            or stored.get("byte_count") not in (None, raw_tag.byte_size)
        ):
            warnings.append("INSPECTION_MAKERNOTE_OFFSET_MISMATCH: bloque distinto")
        return warnings

    def inspect(self) -> dict[str, Any]:
        reader = self.reader if self.reader is not None else TiffReader(self.path).parse()
        exif = reader.get("IFD0/EXIF")
        root = reader.get("IFD0")
        maker_tag = exif.tag(0x927C) if exif else None
        if not maker_tag or not maker_tag.byte_size:
            raise MakerNoteParseError("No se encontró EXIF MakerNote (0x927C)")
        if maker_tag.byte_size > MAX_MAKER_BYTES:
            raise MakerNoteParseError("MakerNote excede límite configurado de lectura")
        offset, length = maker_tag.data_offset, maker_tag.byte_size
        if offset < 0 or offset + length > reader.size:
            raise MakerNoteParseError("Bloque MakerNote sobrepasa límites del NEF")
        with self.path.open("rb") as f:
            f.seek(offset)
            data = f.read(length)
        if len(data) != length:
            raise MakerNoteParseError("MakerNote truncado")
        self.last_makernote_bytes = data
        parsed = MakerNoteIFDReader(data, offset)
        if parsed.kind != "nikon_type3_embedded_tiff":
            raise MakerNoteParseError(f"MakerNote Nikon {parsed.kind}: pendiente soporte")
        main = parsed.ifd(parsed.first_ifd_relative or 0)
        tags: dict[int, MakerNoteTag] = main["tags"]
        get = lambda ident: tags.get(ident)

        def scalar(ident: int) -> Any:
            tag = get(ident)
            return tag.decode(parsed.endian) if tag is not None else None

        make = root.value(0x010F, reader.endian) if root else None
        model = root.value(0x0110, reader.endian) if root else None
        compression_tag = get(0x0093)
        mode = scalar(0x0093)
        if not isinstance(mode, int):
            mode = None
        mode_name = NEF_COMPRESSION.get(mode, "unknown")

        raw = NefInspector(self.path, reader=reader).inspect()["raw"]
        prior_raw_warnings: list[str] = []
        if self.inspection is not None:
            previous_raw = self.inspection.get("raw") or {}
            for fld in ("width", "height", "cfa", "bits_per_sample", "compression"):
                if previous_raw.get(fld) != (raw or {}).get(fld):
                    prior_raw_warnings.append(f"INSPECTION_RAW_MISMATCH_{fld.upper()}: JSON y NEF no coinciden")
        tiff_comp = (raw or {}).get("compression", {}).get("code")
        comp_consistent = (
            (
                (mode in (1, 3, 4) and tiff_comp == 34713)
                or (mode in (2, 5, 6, 7, 9, 10) and tiff_comp in (1, 32769, 34713))
            )
            if mode is not None
            else None
        )
        warnings = list(parsed.warnings) + main["warnings"] + prior_raw_warnings
        warnings.extend(self._validate_inspection(self.inspection, self.path, maker_tag, make, model))
        if comp_consistent is False:
            warnings.append(f"COMPRESSION_NOT_TYPICAL: MakerNote={mode}, TIFF={tiff_comp}")
        if mode is None:
            warnings.append("NEF_COMPRESSION_TAG_MISSING_OR_INVALID")

        linearization_tag = get(0x0096)
        linearization = {
            "present": linearization_tag is not None,
            "decoded": False,
            "byte_count": linearization_tag.value_byte_count if linearization_tag else None,
            "source": self._provenance(linearization_tag),
            "note": "Bloque propietario necesario para ciertas variantes del códec; no es aún una LUT decodificada.",
        }
        raw_center = scalar(0x0099)
        if not (isinstance(raw_center, list) and len(raw_center) == 2 and all(isinstance(i, int) for i in raw_center)):
            raw_center = None
        pixel_size = scalar(0x009A)
        if not (isinstance(pixel_size, list) and len(pixel_size) == 2):
            pixel_size = None
        color_tag = get(0x0097)
        lens_tag = get(0x0098)
        maker_version_tag = get(0x0001)
        maker_version = (
            _ascii_version(maker_version_tag.value) if maker_version_tag and maker_version_tag.value else None
        )
        wb_tag = get(0x000C)
        wb_rb = scalar(0x000C)
        if not (isinstance(wb_rb, list) and len(wb_rb) == 4):
            wb_rb = None

        sha = (
            self.precomputed_sha256
            if self.precomputed_sha256 is not None
            else (sha256_file(self.path) if self.compute_sha256 else None)
        )
        source = {
            "filename": self.path.name,
            "byte_count": reader.size,
            "sha256": sha,
        }
        if sha is not None:
            stored_hash = (self.inspection or {}).get("source", {}).get("sha256")
            if stored_hash and stored_hash.lower() != sha:
                warnings.append("INSPECTION_SHA256_MISMATCH: el fichero no coincide con el informe previo")

        nested: list[dict[str, Any]] = []
        ptr = scalar(0x0011)
        if isinstance(ptr, int) and ptr > 0:
            try:
                preview = parsed.ifd(ptr, depth=1)
                preview_offset_tag = preview["tags"].get(0x0201)
                preview_length_tag = preview["tags"].get(0x0202)
                jpeg_rel = preview_offset_tag.decode(parsed.endian) if preview_offset_tag else None
                jpeg_size = preview_length_tag.decode(parsed.endian) if preview_length_tag else None
                jpeg_abs = None
                jpeg_inside = False
                jpeg_signature = None
                jpeg_dims = None
                if (
                    isinstance(jpeg_rel, int)
                    and isinstance(jpeg_size, int)
                    and jpeg_rel >= 0
                    and jpeg_size > 0
                    and parsed.tiff_relative_base is not None
                ):
                    jpeg_abs = offset + parsed.tiff_relative_base + jpeg_rel
                    jpeg_inside = parsed.tiff_relative_base + jpeg_rel + jpeg_size <= len(data)
                    if jpeg_inside:
                        jpeg_signature = (
                            data[parsed.tiff_relative_base + jpeg_rel : parsed.tiff_relative_base + jpeg_rel + 2]
                            == b"\xff\xd8"
                        )
                        if jpeg_signature:
                            jpeg_dims = _jpeg_sof_dimensions(self.path, jpeg_abs, jpeg_size)
                nested.append(
                    {
                        "kind": "NikonPreviewIFD",
                        "pointer_source_tag": "0x0011",
                        "ifd_relative_offset": ptr,
                        "ifd_absolute_offset": preview["ifd_absolute_offset"],
                        "tag_count": preview["tag_count"],
                        "jpeg_preview": {
                            "data_offset": jpeg_abs,
                            "byte_count": jpeg_size,
                            "inside_makernote": jpeg_inside,
                            "has_jpeg_soi": jpeg_signature,
                            "width": jpeg_dims[0] if jpeg_dims else None,
                            "height": jpeg_dims[1] if jpeg_dims else None,
                        },
                        "tags": [
                            t.to_json(parsed.endian, names=PREVIEW_NAMES, show_sensitive=self.show_sensitive)
                            for t in preview["tags"].values()
                        ],
                        "warnings": preview["warnings"],
                    }
                )
            except MakerNoteParseError as exc:
                warnings.append(f"NikonPreviewIFD invalid: {exc}")

        if mode in (1, 3, 4):
            patch_status = "encoder_required"
            reason = "Huffman Nikon comprimido: no se admite sustitución de muestras sin codificación exacta."
        elif mode in (2, 5, 6, 7, 9, 10):
            patch_status = "layout_validation_required"
            reason = "Modo no Huffman según etiqueta; deben validarse empaquetamiento y tamaños antes de escribir."
        else:
            patch_status = "unknown_mode"
            reason = "No se pudo identificar un modo de codificación suficientemente fiable."

        return {
            "schema_version": SCHEMA_VERSION,
            "tool": {
                "name": "nikon_makernote_inspector",
                "version": "0.2.0",
                "read_only": True,
                "raw_pixels_decoded": False,
                "private_substructures_decoded": False,
            },
            "source": source,
            "camera": {"make": make, "model": model},
            "maker_note": {
                "variant": parsed.kind,
                "header_version": parsed.version,
                "nikon_main_version": maker_version,
                "file_offset": offset,
                "byte_count": length,
                "tiff_base_file_offset": offset + (parsed.tiff_relative_base or 0),
                "tiff_byte_order": "big_endian_MM" if parsed.endian == ">" else "little_endian_II",
                "first_ifd_relative_offset": parsed.first_ifd_relative,
                "main_ifd_file_offset": main["ifd_absolute_offset"],
                "main_ifd_tag_count": main["tag_count"],
                "tags": [t.to_json(parsed.endian, show_sensitive=self.show_sensitive) for t in tags.values()],
                "nested_ifds": nested,
            },
            "nef_compression": {
                "tiff_compression_code": tiff_comp,
                "maker_note_tag_id": "0x0093",
                "maker_note_mode_code": mode,
                "maker_note_mode_name": mode_name,
                "is_lossless": (mode == 3 or mode in (2, 5, 6, 7, 9, 10)) if mode is not None else None,
                "is_lossy": mode in (1, 4) if mode is not None else None,
                "tiff_makernote_codes_plausibly_consistent": comp_consistent,
                "source": self._provenance(compression_tag),
                "direct_binary_replacement_supported": False,
                "injection_status": patch_status,
                "injection_reason": reason,
            },
            "sensor": {
                "raw_image_center": {
                    "coordinates": raw_center,
                    "units": "RAW pixel indices",
                    "source": self._provenance(get(0x0099)),
                },
                "sensor_pixel_size": {
                    "values": pixel_size,
                    "unit": "not_assumed",
                    "source": self._provenance(get(0x009A)),
                },
                "black_level": {"value": None, "status": "not_available_from_decoded_tags"},
                "white_level": {"value": None, "status": "not_available_from_decoded_tags"},
                "raw_cfa": (raw or {}).get("cfa"),
                "raw_dimensions": {
                    "width": (raw or {}).get("width"),
                    "height": (raw or {}).get("height"),
                },
            },
            "white_balance": {
                "mode": scalar(0x0007),
                "wb_rb_levels": wb_rb,
                "wb_rb_levels_source": self._provenance(wb_tag),
                "color_balance_block": {
                    "present": color_tag is not None,
                    "byte_count": color_tag.value_byte_count if color_tag else None,
                    "version_prefix": _ascii_version(color_tag.value) if color_tag and color_tag.value else None,
                    "decoded": False,
                    "encrypted_or_vendor_specific": True,
                    "source": self._provenance(color_tag),
                },
                "note": "WB_RBLevels no equivale a una matriz de color ni a ganancias lineales sin calibración.",
            },
            "proprietary_blocks": {
                "linearization_table": linearization,
                "lens_data": {
                    "present": lens_tag is not None,
                    "decoded": False,
                    "byte_count": lens_tag.value_byte_count if lens_tag else None,
                    "version_prefix": _ascii_version(lens_tag.value) if lens_tag and lens_tag.value else None,
                    "source": self._provenance(lens_tag),
                },
            },
            "validation": {
                "maker_note_within_nef": True,
                "tags_out_of_bounds": [f"0x{tag.id:04X}" for tag in tags.values() if not tag.offset_valid],
                "existing_inspection_supplied": self.inspection is not None,
                "exif_makernote_tag_source": {
                    "ifd_path": exif.path if exif else None,
                    "ifd_entry_offset": maker_tag.entry_offset,
                },
                "warnings": warnings,
            },
        }


# ---------------------------------------------------------------------------
# Contexto Nikon 14-bit Lossless y Cabecera 0x0096
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LosslessHeader:
    file_offset: int
    byte_count: int
    version_hex: str
    seed_parities: tuple[tuple[int, int], tuple[int, int]]
    curve_size_field: int
    curve_mode: str
    bit_depth: int
    huffman_table: str
    supported: bool

    @classmethod
    def from_bytes(cls, data: bytes, offset: int, *, endian: str) -> "LosslessHeader":
        if len(data) < 12:
            raise CodecError("Cabecera Nikon 0x0096 incompleta")
        if endian not in (">", "<"):
            raise CodecError("Orden de bytes inválido")
        ver0, ver1 = data[:2]
        seeds = struct.unpack_from(endian + "4H", data, 2)
        csize = struct.unpack_from(endian + "H", data, 10)[0]
        supported = (ver0, ver1) == (0x46, 0x30)
        return cls(
            offset,
            len(data),
            data[:2].hex(),
            ((seeds[0], seeds[1]), (seeds[2], seeds[3])),
            csize,
            "identity_0x46" if ver0 == 0x46 else "unknown_or_nonlinear",
            14,
            "fixed_14bit_lossless",
            supported,
        )


@dataclass(frozen=True)
class NefLosslessContext:
    nef_path: Path
    nef_sha256: str
    manufacturer: str
    model: str
    width: int
    height: int
    raw_offset: int
    raw_length: int
    raw_count: int
    strip_byte_count_field_offset: int
    endian: str
    header: LosslessHeader

    @classmethod
    def from_reports(
        cls,
        nef: str | Path,
        j1: dict[str, Any],
        j2: dict[str, Any],
        *,
        hash_check: bool = True,
        precomputed_sha256: str | None = None,
    ) -> "NefLosslessContext":
        p = Path(nef)
        if not p.is_file():
            raise CodecError(f"NEF inexistente: {p}")
        size = p.stat().st_size
        sources = (j1.get("source", {}), j2.get("source", {}))
        for source in sources:
            if source.get("byte_count") != size:
                raise CodecError("El tamaño del NEF no coincide con los informes anteriores")
        digest = (
            precomputed_sha256
            if precomputed_sha256 is not None
            else (sha256_file(p) if hash_check else "not_computed")
        )
        if hash_check:
            for source in sources:
                if source.get("sha256") and source["sha256"].lower() != digest:
                    raise CodecError("SHA-256 distinto: posible combinación de NEF e informes no correspondientes")
        raw = j1.get("raw") or {}
        mk = j2.get("nef_compression") or {}
        if j1.get("camera", {}).get("model") != j2.get("camera", {}).get("model"):
            raise CodecError("Modelo incompatible entre etapas")
        if j1.get("camera", {}).get("model") != "NIKON D610":
            raise CodecError("Códec validado únicamente para Nikon D610")
        if raw.get("compression", {}).get("code") != 34713 or mk.get("maker_note_mode_code") != 3:
            raise CodecError("Requiere compresión Nikon 34713 y modalidad MakerNote 3 (lossless)")
        if raw.get("bits_per_sample") != [14] or raw.get("samples_per_pixel") != 1:
            raise CodecError("Requiere RAW Bayer plano de 14 bits")
        if raw.get("cfa", {}).get("pattern") != "RGGB":
            raise CodecError("CFA no admitido en esta etapa")
        segs = raw.get("segments") or []
        if len(segs) != 1 or segs[0].get("kind") != "strip" or not segs[0].get("inside_file"):
            raise CodecError("Solo se admiten NEF con un único strip RAW válido")
        seg = segs[0]
        offset, length = int(seg["offset"]), int(seg["byte_count"])
        if offset < 8 or length <= 0 or offset + length > size:
            raise CodecError("Bloque de datos RAW fuera del archivo")
        lin = j2.get("proprietary_blocks", {}).get("linearization_table", {})
        loc = (lin.get("source") or {}).get("data_offset")
        nbytes = lin.get("byte_count")
        if not isinstance(loc, int) or not isinstance(nbytes, int) or nbytes < 12:
            raise CodecError("Etiqueta 0x0096 no localizada")
        if loc < 8 or loc + nbytes > size:
            raise CodecError("Rango 0x0096 inválido")
        endian = {"big_endian_MM": ">", "little_endian_II": "<"}.get(
            j1.get("format", {}).get("byte_order")
        )
        if endian is None:
            raise CodecError("Orden de bytes TIFF no reconocido")
        with p.open("rb") as fd:
            fd.seek(loc)
            head = LosslessHeader.from_bytes(fd.read(nbytes), loc, endian=endian)
        if not head.supported or head.curve_mode != "identity_0x46":
            raise CodecError("Versión del encabezamiento Nikon no implementada")
        if max(v for row in head.seed_parities for v in row) >= 1 << 14:
            raise CodecError("Semillas predictivas fuera de rango")
        matches = [
            ifd for ifd in j1.get("tiff", {}).get("ifds", []) if ifd.get("path") == raw.get("ifd_path")
        ]
        if len(matches) != 1:
            raise CodecError("No se ha podido identificar el IFD RAW original")
        tags = [t for t in matches[0].get("tags", []) if t.get("id") == "0x0117"]
        if len(tags) != 1 or tags[0].get("value") != length or tags[0].get("type_id") != 4:
            raise CodecError("Etiqueta StripByteCounts ambigua o incorrecta")
        field_off = tags[0]["value_data_offset"]
        with p.open("rb") as fd:
            fd.seek(field_off)
            actual = struct.unpack(endian + "I", fd.read(4))[0]
        if actual != length:
            raise CodecError("ByteCount del TIFF físico diferente al JSON")
        w, h = int(raw["width"]), int(raw["height"])
        if w < 2 or h < 2 or w % 2 or h % 2 or w * h > 160_000_000:
            raise CodecError("Dimensiones inesperadas del RAW")
        return cls(
            p, digest, j1["camera"]["make"], j1["camera"]["model"],
            w, h, offset, length, w * h, field_off, endian, head,
        )

    @classmethod
    def open(
        cls,
        nef: str | Path,
        inspection: str | Path | dict[str, Any],
        makernote: str | Path | dict[str, Any],
        *,
        hash_check: bool = True,
        precomputed_sha256: str | None = None,
    ) -> "NefLosslessContext":
        j1 = inspection if isinstance(inspection, dict) else read_json(inspection)
        j2 = makernote if isinstance(makernote, dict) else read_json(makernote)
        return cls.from_reports(nef, j1, j2, hash_check=hash_check, precomputed_sha256=precomputed_sha256)


# ---------------------------------------------------------------------------
# Parcheador Transaccional del Contenedor NEF
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchField:
    tag: str
    location: int
    original: int
    replacement: int

    def as_report(self) -> dict[str, Any]:
        return asdict(self)


def _to_nikon_tiff_ep_jpeg(data: bytes) -> bytes:
    """Normaliza un flujo JPEG Baseline al formato TIFF/EP del codificador Nikon D610.

    - Elimina el marcador APP0/JFIF (0xFFE0) añadido por Pillow.
    - Unifica las tablas DQT (0xFFDB) en un único segmento de 132 bytes.
    - Mantiene SOF0 (0xFFC0) con submuestreo cromático 4:2:2 (2x1).
    - Unifica las tablas Huffman estándar DHT (0xFFC4) en un único segmento de 418 bytes.
    """
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise InjectionError("Flujo JPEG inválido al normalizar a TIFF/EP")
    dqt_payloads: list[bytes] = []
    dht_payloads: list[bytes] = []
    sof0_seg: bytes | None = None
    other_segs: list[bytes] = []
    sos_and_tail: bytes | None = None
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise InjectionError("Marcador JPEG desalineado")
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1
        if marker == 0xD9:
            break
        if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
            continue
        if i + 2 > len(data):
            raise InjectionError("Cabecera de marcador JPEG truncada")
        length = struct.unpack(">H", data[i : i + 2])[0]
        if length < 2 or i + length > len(data):
            raise InjectionError("Longitud de segmento JPEG fuera de rango")
        seg_body = data[i + 2 : i + length]
        if marker == 0xDA:
            sos_and_tail = b"\xff\xda" + data[i:]
            break
        if marker == 0xE0:
            pass  # Omitir APP0/JFIF para compatibilidad con el visor hardware de cámara Nikon
        elif marker == 0xDB:
            dqt_payloads.append(seg_body)
        elif marker == 0xC4:
            dht_payloads.append(seg_body)
        elif marker == 0xC0:
            sof0_seg = b"\xff\xc0" + data[i : i + length]
        else:
            other_segs.append(bytes([0xFF, marker]) + data[i : i + length])
        i += length
    if not dqt_payloads or not dht_payloads or sof0_seg is None or sos_and_tail is None:
        return data
    dqt_body = b"".join(dqt_payloads)
    dht_body = b"".join(dht_payloads)
    out = bytearray(b"\xff\xd8")
    out += b"\xff\xdb" + struct.pack(">H", len(dqt_body) + 2) + dqt_body
    out += sof0_seg
    out += b"\xff\xc4" + struct.pack(">H", len(dht_body) + 2) + dht_body
    for seg in other_segs:
        out += seg
    out += sos_and_tail
    return bytes(out)


@dataclass(frozen=True)
class PreviewSpec:
    """Contrato de una vista previa TIFF o MakerNote reconocida de la plantilla."""

    ifd_path: str
    storage: str
    width: int
    height: int
    offset_tag: str
    length_tag: str
    original_offset: int = 0
    original_length: int = 0
    in_place: bool = False
    relative_base_offset: int = 0


@dataclass(frozen=True)
class PreviewPayload:
    """Bytes regenerados para una vista previa reconocida."""

    ifd_path: str
    storage: str
    width: int
    height: int
    data: bytes

    def as_report(self) -> dict[str, Any]:
        return {
            "ifd_path": self.ifd_path,
            "storage": self.storage,
            "width": self.width,
            "height": self.height,
            "bytes": len(self.data),
            "sha256": sha256_bytes(self.data),
        }


@dataclass(frozen=True)
class PreviewBlock:
    """Ubicación final de un preview regenerado dentro del nuevo NEF."""

    ifd_path: str
    storage: str
    width: int
    height: int
    offset: int
    byte_count: int
    sha256: str
    in_place: bool = False
    original_offset: int = 0
    original_length: int = 0

    def as_report(self) -> dict[str, Any]:
        return {
            "ifd_path": self.ifd_path,
            "storage": self.storage,
            "width": self.width,
            "height": self.height,
            "offset": self.offset,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "in_place": self.in_place,
        }


@dataclass(frozen=True)
class InjectionPlan:
    strategy: str
    source_size: int
    original_payload_offset: int
    original_payload_bytes: int
    payload_offset: int
    payload_bytes: int
    output_size: int
    patches: tuple[PatchField, ...]
    raw_ifd: str
    # Se añaden al final para que los callers que construían el plan posicional
    # anterior sigan representando un contenedor sin previews regenerados.
    prefix_bytes: int | None = None
    previews: tuple[PreviewBlock, ...] = ()
    preview_padding_bytes: int = 0

    def as_report(self) -> dict[str, Any]:
        prefix_bytes = self.prefix_bytes if self.prefix_bytes is not None else self.payload_offset
        return {
            "strategy": self.strategy,
            "source_size": self.source_size,
            "original_payload": {
                "offset": self.original_payload_offset,
                "bytes": self.original_payload_bytes,
            },
            "prefix_bytes": prefix_bytes,
            "embedded_previews": [preview.as_report() for preview in self.previews],
            "preview_padding_bytes": self.preview_padding_bytes,
            "new_payload": {
                "offset": self.payload_offset,
                "bytes": self.payload_bytes,
            },
            "output_size": self.output_size,
            "raw_ifd": self.raw_ifd,
            "patches": [p.as_report() for p in self.patches],
        }


class NefContainerPatcher:
    """Planifica, construye y verifica en memoria/disco un parche NEF TIFF clásico."""

    def __init__(
        self,
        ctx: NefLosslessContext,
        inspection: dict[str, Any],
        makernote: dict[str, Any],
        *,
        cached_prefix_bytes: bytes | None = None,
    ) -> None:
        self.ctx = ctx
        self.inspection = inspection
        self.makernote = makernote
        self._cached_prefix_bytes = cached_prefix_bytes

    def _ifd(self, path: str) -> dict[str, Any]:
        matches = [
            item
            for item in self.inspection.get("tiff", {}).get("ifds", [])
            if item.get("path") == path
        ]
        if len(matches) != 1:
            raise InjectionError(f"No se encuentra un IFD único para {path}")
        return matches[0]

    @staticmethod
    def _tag_map(ifd: dict[str, Any], *ids: str) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for tag in ifd.get("tags", []):
            key = tag.get("id")
            if key in ids:
                if key in found:
                    raise InjectionError(f"Etiqueta TIFF duplicada: {key}")
                found[key] = tag
        if set(found) != set(ids):
            missing = ", ".join(sorted(set(ids) - set(found)))
            raise InjectionError(f"Faltan etiquetas TIFF requeridas: {missing}")
        return found

    def preview_specs(self) -> tuple[PreviewSpec, ...]:
        """Devuelve los previews que pueden regenerarse sin cambiar IFDs.

        Se admiten el thumbnail RGB de 8 bits de un strip, los JPEG definidos por
        JPEGInterchangeFormat en IFDs TIFF y el NikonPreviewIFD embebido en el MakerNote
        (tag 0x0011). No se deja ninguna vista previa antigua silenciosamente:
        cualquier formato ambiguo aborta la inyección antes de modificar archivos.
        """
        specs: list[PreviewSpec] = []
        seen_paths: set[str] = set()
        for preview in self.inspection.get("previews", []):
            ifd_path = preview.get("ifd_path")
            width, height = preview.get("width"), preview.get("height")
            if not isinstance(ifd_path, str) or not ifd_path:
                raise InjectionError("Preview sin ruta IFD identificable")
            if ifd_path in seen_paths:
                raise InjectionError(f"Preview duplicado en {ifd_path}")
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                raise InjectionError(f"Preview {ifd_path} sin dimensiones válidas")
            if width * height > 160_000_000:
                raise InjectionError(f"Preview {ifd_path} excesivamente grande")
            segments = preview.get("segments") or []
            if len(segments) != 1 or not segments[0].get("inside_file"):
                raise InjectionError(
                    f"Preview {ifd_path} requiere exactamente un segmento dentro del archivo"
                )
            segment = segments[0]
            orig_off = int(segment["offset"])
            orig_len = int(segment["byte_count"])
            if segment.get("kind") == "jpeg_preview":
                specs.append(
                    PreviewSpec(
                        ifd_path,
                        "jpeg",
                        width,
                        height,
                        "0x0201",
                        "0x0202",
                        original_offset=orig_off,
                        original_length=orig_len,
                        in_place=False,
                        relative_base_offset=0,
                    )
                )
            elif segment.get("kind") == "strip":
                bits = preview.get("bits_per_sample")
                if (
                    preview.get("compression", {}).get("code") != 1
                    or preview.get("photometric", {}).get("code") != 2
                    or preview.get("samples_per_pixel") != 3
                    or bits != [8, 8, 8]
                    or preview.get("rows_per_strip") != height
                    or orig_len != width * height * 3
                ):
                    raise InjectionError(
                        f"Preview RGB {ifd_path} no usa un único strip RGB8 sin comprimir"
                    )
                specs.append(
                    PreviewSpec(
                        ifd_path,
                        "rgb8_strip",
                        width,
                        height,
                        "0x0111",
                        "0x0117",
                        original_offset=orig_off,
                        original_length=orig_len,
                        in_place=True,
                        relative_base_offset=0,
                    )
                )
            else:
                raise InjectionError(f"Formato de preview no admitido en {ifd_path}")
            seen_paths.add(ifd_path)

        mk = self.makernote.get("maker_note") or {}
        tiff_base = int(mk.get("tiff_base_file_offset") or 0)
        for nested in mk.get("nested_ifds", []):
            if nested.get("kind") != "NikonPreviewIFD":
                continue
            ifd_path = "IFD0/EXIF/MakerNote/NikonPreviewIFD"
            if ifd_path in seen_paths:
                raise InjectionError(f"Preview MakerNote duplicado en {ifd_path}")
            jp = nested.get("jpeg_preview") or {}
            orig_off = jp.get("data_offset")
            orig_len = jp.get("byte_count")
            if (
                not isinstance(orig_off, int)
                or not isinstance(orig_len, int)
                or orig_off <= 0
                or orig_len <= 0
                or not jp.get("inside_makernote")
                or not jp.get("has_jpeg_soi")
            ):
                raise InjectionError(f"Preview MakerNote {ifd_path} inválido o fuera de rango")
            width, height = jp.get("width"), jp.get("height")
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                dims = _jpeg_sof_dimensions(self.ctx.nef_path, orig_off, orig_len)
                if not dims:
                    raise InjectionError(f"No se pudieron determinar dimensiones SOF0 de {ifd_path}")
                width, height = dims
            specs.append(
                PreviewSpec(
                    ifd_path,
                    "jpeg",
                    width,
                    height,
                    "0x0201",
                    "0x0202",
                    original_offset=orig_off,
                    original_length=orig_len,
                    in_place=True,
                    relative_base_offset=tiff_base,
                )
            )
            seen_paths.add(ifd_path)
        return tuple(specs)

    @staticmethod
    def _cover_resize_rgb8(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise InjectionError("El preview fuente debe ser RGB uint8 con tres canales")
        source_height, source_width = rgb.shape[:2]
        if source_width <= 0 or source_height <= 0:
            raise InjectionError("El preview fuente no puede estar vacío")
        scale = max(width / source_width, height / source_height)
        resized_width = max(width, round(source_width * scale))
        resized_height = max(height, round(source_height * scale))
        image = Image.fromarray(rgb, mode="RGB")
        if (resized_width, resized_height) != (source_width, source_height):
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        crop_x = (resized_width - width) // 2
        crop_y = (resized_height - height) // 2
        return np.asarray(image.crop((crop_x, crop_y, crop_x + width, crop_y + height)))

    @staticmethod
    def _encode_nikon_jpeg(
        image: np.ndarray,
        *,
        max_bytes: int | None = None,
        strict_fit: bool = False,
    ) -> bytes:
        pil_img = Image.fromarray(image, mode="RGB")
        qualities = (
            (92, 88, 85, 80, 75, 70, 65, 60, 50, 40, 30, 20, 10, 5, 1)
            if strict_fit
            else ((92, 88, 85, 80, 75, 70) if (max_bytes and max_bytes > 0) else (92,))
        )
        last_data = b""
        for q in qualities:
            stream = io.BytesIO()
            pil_img.save(
                stream,
                format="JPEG",
                quality=q,
                subsampling=1,  # 4:2:2 horizontal (2x1), requerido por el visor de cámara Nikon
                optimize=False,
                progressive=False,
            )
            candidate = _to_nikon_tiff_ep_jpeg(stream.getvalue())
            last_data = candidate
            if max_bytes is None or max_bytes <= 0 or len(candidate) <= max_bytes:
                return candidate
        if strict_fit and max_bytes and len(last_data) > max_bytes:
            raise InjectionError(
                f"El preview JPEG ({len(last_data)} B) excede la capacidad fija del bloque ({max_bytes} B)"
            )
        stream = io.BytesIO()
        pil_img.save(
            stream,
            format="JPEG",
            quality=85,
            subsampling=1,
            optimize=False,
            progressive=False,
        )
        return _to_nikon_tiff_ep_jpeg(stream.getvalue())

    def build_preview_payloads(self, raw_rgb8: np.ndarray) -> tuple[PreviewPayload, ...]:
        """Codifica cada preview de la plantilla a partir del RGB ya ajustado al RAW."""
        expected_shape = (self.ctx.height, self.ctx.width, 3)
        if raw_rgb8.dtype != np.uint8 or raw_rgb8.shape != expected_shape:
            raise InjectionError(
                f"Preview fuente con forma {raw_rgb8.shape}; se esperaba RGB8 {expected_shape}"
            )
        payloads: list[PreviewPayload] = []
        for spec in self.preview_specs():
            image = self._cover_resize_rgb8(raw_rgb8, spec.width, spec.height)
            if spec.storage == "rgb8_strip":
                data = np.ascontiguousarray(image).tobytes()
            elif spec.storage == "jpeg":
                data = self._encode_nikon_jpeg(
                    image,
                    max_bytes=spec.original_length if spec.original_length > 0 else None,
                    strict_fit=spec.in_place,
                )
                with Image.open(io.BytesIO(data)) as decoded:
                    if decoded.format != "JPEG" or decoded.size != (spec.width, spec.height):
                        raise InjectionError(f"No se pudo validar el JPEG regenerado de {spec.ifd_path}")
            else:  # defensa ante contratos construidos fuera de preview_specs
                raise InjectionError(f"Almacenamiento de preview desconocido: {spec.storage}")
            if not data or len(data) > MAX_UINT32:
                raise InjectionError(f"Longitud de preview inválida en {spec.ifd_path}")
            payloads.append(
                PreviewPayload(spec.ifd_path, spec.storage, spec.width, spec.height, data)
            )
        return tuple(payloads)

    def _ordered_preview_payloads(
        self,
        preview_payloads: Sequence[PreviewPayload],
    ) -> tuple[PreviewPayload, ...]:
        specs = self.preview_specs()
        if not preview_payloads:
            if specs:
                raise InjectionError(
                    "La plantilla contiene previews; se requieren payloads regenerados para no conservarlos obsoletos"
                )
            return ()
        by_path = {payload.ifd_path: payload for payload in preview_payloads}
        if len(by_path) != len(preview_payloads) or set(by_path) != {spec.ifd_path for spec in specs}:
            raise InjectionError("Los previews entregados no coinciden exactamente con los IFD de la plantilla")
        ordered: list[PreviewPayload] = []
        for spec in specs:
            payload = by_path[spec.ifd_path]
            if (
                payload.storage != spec.storage
                or payload.width != spec.width
                or payload.height != spec.height
                or not isinstance(payload.data, bytes)
                or not payload.data
            ):
                raise InjectionError(f"Preview regenerado incompatible con {spec.ifd_path}")
            if spec.storage == "rgb8_strip" and len(payload.data) != spec.width * spec.height * 3:
                raise InjectionError(f"Longitud RGB8 inválida para {spec.ifd_path}")
            ordered.append(payload)
        return tuple(ordered)

    def _validated_patch(
        self,
        fd: BinaryIO,
        *,
        ifd_path: str,
        tag_id: str,
        tag: dict[str, Any],
        original: int,
        replacement: int,
    ) -> PatchField:
        if tag.get("type_id") != 4 or tag.get("count") != 1 or not tag.get("value_is_inline"):
            raise InjectionError(f"{ifd_path}:{tag_id} no es un LONG escalar inline")
        physical = int(tag["value_field_offset"])
        if physical < 8 or physical + 4 > self.ctx.raw_offset:
            raise InjectionError(f"Campo {ifd_path}:{tag_id} fuera de cabeceras TIFF")
        if int(tag["value"]) != original:
            raise InjectionError(f"Valor reportado {ifd_path}:{tag_id} distinto al esperado")
        fd.seek(physical)
        raw = fd.read(4)
        if len(raw) != 4 or struct.unpack(self.ctx.endian + "I", raw)[0] != original:
            raise InjectionError(f"Valor físico de {ifd_path}:{tag_id} distinto al JSON")
        return PatchField(f"{ifd_path}:{tag_id}", physical, original, replacement)

    def _raw_tags(self) -> dict[str, dict[str, Any]]:
        raw = self.inspection.get("raw") or {}
        ifds = [
            it
            for it in self.inspection.get("tiff", {}).get("ifds", [])
            if it.get("path") == raw.get("ifd_path")
        ]
        if len(ifds) != 1:
            raise InjectionError("No se encuentra un IFD RAW único")
        found: dict[str, dict[str, Any]] = {}
        for t in ifds[0].get("tags", []):
            key = t.get("id")
            if key in ("0x0111", "0x0117"):
                if key in found:
                    raise InjectionError(f"Etiqueta TIFF duplicada: {key}")
                found[key] = t
        if set(found) != {"0x0111", "0x0117"}:
            raise InjectionError("Faltan StripOffsets o StripByteCounts")
        return found

    def _safe_to_replace_tail(self) -> None:
        ctx = self.ctx
        if ctx.raw_offset + ctx.raw_length != ctx.nef_path.stat().st_size:
            raise InjectionError("replace-tail exige que el strip RAW termine exactamente en EOF")
        for it in self.inspection.get("tiff", {}).get("ifds", []):
            ifd_location = it.get("ifd_offset")
            if not isinstance(ifd_location, int) or ifd_location >= ctx.raw_offset:
                raise InjectionError("IFD situado dentro de la zona que se pretende sustituir")
            ifd_end = ifd_location + 2 + 12 * int(it.get("tag_count", 0)) + 4
            if ifd_end > ctx.raw_offset:
                raise InjectionError("El directorio TIFF se solapa con el strip RAW")
            for tag in it.get("tags", []):
                offset = tag.get("value_data_offset")
                size = tag.get("value_byte_count")
                if isinstance(offset, int) and isinstance(size, int):
                    if offset < 0 or size < 0 or offset + size > ctx.raw_offset:
                        raise InjectionError("Un valor TIFF distinto del strip está al final del archivo")
        for preview in self.inspection.get("previews", []):
            for segment in preview.get("segments", []):
                if int(segment["offset"]) + int(segment["byte_count"]) > ctx.raw_offset:
                    raise InjectionError("Vista previa JPEG ubicada en la zona del strip RAW")
        mk = self.makernote.get("maker_note", {})
        if not (
            0 <= int(mk.get("file_offset", -1))
            and int(mk.get("file_offset", -1)) + int(mk.get("byte_count", 0)) <= ctx.raw_offset
        ):
            raise InjectionError("MakerNote solapado con el strip RAW")

    def _makernote_preview_ifd(self) -> dict[str, Any]:
        mk = self.makernote.get("maker_note") or {}
        matches = [
            item
            for item in mk.get("nested_ifds", [])
            if item.get("kind") == "NikonPreviewIFD"
        ]
        if len(matches) != 1:
            raise InjectionError("No se encuentra un NikonPreviewIFD único en MakerNote")
        return matches[0]

    def plan(
        self,
        payload_bytes: int,
        *,
        strategy: str = "replace-tail",
        preview_payloads: Sequence[PreviewPayload] = (),
    ) -> InjectionPlan:
        """Planifica previews en su hueco nativo (o antes del RAW si desbordan) manteniendo el RAW en EOF."""
        ctx = self.ctx
        if payload_bytes <= 0 or payload_bytes > MAX_UINT32:
            raise InjectionError("Longitud de payload incompatible con TIFF clásico")
        if strategy not in ("append", "replace-tail"):
            raise InjectionError("Estrategia de inyección desconocida")
        size = ctx.nef_path.stat().st_size
        if strategy == "replace-tail":
            self._safe_to_replace_tail()

        ordered_previews = self._ordered_preview_payloads(preview_payloads)
        preview_specs = {spec.ifd_path: spec for spec in self.preview_specs()}
        prefix_bytes = size if strategy == "append" else ctx.raw_offset
        cursor = prefix_bytes
        blocks: list[PreviewBlock] = []
        has_trailing_previews = False
        for payload in ordered_previews:
            spec = preview_specs[payload.ifd_path]
            byte_count = len(payload.data)
            can_place_in_place = (
                spec.original_offset > 0
                and spec.original_length > 0
                and spec.original_offset + spec.original_length <= prefix_bytes
                and byte_count <= spec.original_length
                and (spec.in_place or strategy == "replace-tail")
            )
            if spec.in_place and not can_place_in_place:
                raise InjectionError(
                    f"El preview in-place {payload.ifd_path} ({byte_count} B) no cabe en su región ({spec.original_length} B)"
                )
            if can_place_in_place:
                blocks.append(
                    PreviewBlock(
                        payload.ifd_path,
                        payload.storage,
                        payload.width,
                        payload.height,
                        spec.original_offset,
                        byte_count,
                        sha256_bytes(payload.data),
                        in_place=True,
                        original_offset=spec.original_offset,
                        original_length=spec.original_length,
                    )
                )
            else:
                if cursor > MAX_UINT32 or byte_count > MAX_UINT32 - cursor:
                    raise InjectionError("Los previews superan la capacidad de TIFF clásico")
                blocks.append(
                    PreviewBlock(
                        payload.ifd_path,
                        payload.storage,
                        payload.width,
                        payload.height,
                        cursor,
                        byte_count,
                        sha256_bytes(payload.data),
                        in_place=False,
                        original_offset=spec.original_offset,
                        original_length=spec.original_length,
                    )
                )
                cursor += byte_count
                has_trailing_previews = True

        preview_padding_bytes = (-cursor) % 512 if has_trailing_previews else 0
        new_offset = cursor + preview_padding_bytes
        new_size = new_offset + payload_bytes
        if new_offset > MAX_UINT32 or new_size > MAX_UINT32:
            raise InjectionError("Offset o archivo supera la capacidad de TIFF clásico")

        previews_by_path = {
            preview.get("ifd_path"): preview for preview in self.inspection.get("previews", [])
        }
        raw_ifd = str(self.inspection["raw"]["ifd_path"])
        raw_tags = self._raw_tags()
        patches: list[PatchField] = []
        with ctx.nef_path.open("rb") as fd:
            for block in blocks:
                spec = preview_specs[block.ifd_path]
                if block.ifd_path == "IFD0/EXIF/MakerNote/NikonPreviewIFD":
                    mk_ifd = self._makernote_preview_ifd()
                    tags = self._tag_map(mk_ifd, spec.offset_tag, spec.length_tag)
                    offset_original = int(tags[spec.offset_tag]["value"])
                    length_original = int(tags[spec.length_tag]["value"])
                    replacement_offset = block.offset - spec.relative_base_offset
                    if replacement_offset < 0:
                        raise InjectionError("Offset relativo de NikonPreviewIFD negativo")
                else:
                    preview = previews_by_path.get(block.ifd_path)
                    segments = preview.get("segments") if isinstance(preview, dict) else None
                    if not isinstance(segments, list) or len(segments) != 1:
                        raise InjectionError(f"Preview {block.ifd_path} sin segmento verificable")
                    segment = segments[0]
                    tags = self._tag_map(
                        self._ifd(block.ifd_path), spec.offset_tag, spec.length_tag
                    )
                    offset_original = int(segment["offset"])
                    length_original = int(segment["byte_count"])
                    replacement_offset = block.offset
                patches.append(
                    self._validated_patch(
                        fd,
                        ifd_path=block.ifd_path,
                        tag_id=spec.offset_tag,
                        tag=tags[spec.offset_tag],
                        original=offset_original,
                        replacement=replacement_offset,
                    )
                )
                patches.append(
                    self._validated_patch(
                        fd,
                        ifd_path=block.ifd_path,
                        tag_id=spec.length_tag,
                        tag=tags[spec.length_tag],
                        original=length_original,
                        replacement=block.byte_count,
                    )
                )
            patches.append(
                self._validated_patch(
                    fd,
                    ifd_path=raw_ifd,
                    tag_id="0x0111",
                    tag=raw_tags["0x0111"],
                    original=ctx.raw_offset,
                    replacement=new_offset,
                )
            )
            patches.append(
                self._validated_patch(
                    fd,
                    ifd_path=raw_ifd,
                    tag_id="0x0117",
                    tag=raw_tags["0x0117"],
                    original=ctx.raw_length,
                    replacement=payload_bytes,
                )
            )
        locations = sorted(patch.location for patch in patches)
        if any(current < previous + 4 for previous, current in zip(locations, locations[1:])):
            raise InjectionError("Los campos TIFF a modificar se solapan")
        return InjectionPlan(
            strategy=strategy,
            source_size=size,
            original_payload_offset=ctx.raw_offset,
            original_payload_bytes=ctx.raw_length,
            payload_offset=new_offset,
            payload_bytes=payload_bytes,
            output_size=new_size,
            patches=tuple(patches),
            raw_ifd=raw_ifd,
            prefix_bytes=prefix_bytes,
            previews=tuple(blocks),
            preview_padding_bytes=preview_padding_bytes,
        )

    def build_and_validate_in_memory(
        self,
        payload_data: bytes | np.ndarray,
        payload_sha256: str,
        provisional_nef: Path,
        plan: InjectionPlan,
        *,
        preview_payloads: Sequence[PreviewPayload] = (),
    ) -> tuple[str, dict[str, Any]]:
        """Construye el NEF sobrescribiendo los huecos de preview originales y validando TIFF + MakerNote."""
        payload_mv = (
            memoryview(np.ascontiguousarray(payload_data).view(np.uint8))
            if isinstance(payload_data, np.ndarray)
            else memoryview(payload_data)
        )
        if len(payload_mv) != plan.payload_bytes:
            raise InjectionError("El tamaño del payload en memoria no coincide con el plan")
        ordered_previews = self._ordered_preview_payloads(preview_payloads)
        if len(ordered_previews) != len(plan.previews):
            raise InjectionError("Los previews recibidos no coinciden con el plan de inyección")
        for block, preview in zip(plan.previews, ordered_previews):
            if (
                preview.ifd_path != block.ifd_path
                or len(preview.data) != block.byte_count
                or sha256_bytes(preview.data) != block.sha256
            ):
                raise InjectionError(f"Preview alterado después de planificar: {block.ifd_path}")

        prefix_bytes = plan.prefix_bytes if plan.prefix_bytes is not None else plan.payload_offset
        if (
            self._cached_prefix_bytes is not None
            and len(self._cached_prefix_bytes) == prefix_bytes
        ):
            orig_prefix = self._cached_prefix_bytes
        else:
            with self.ctx.nef_path.open("rb") as src:
                orig_prefix = src.read(prefix_bytes)
            if len(orig_prefix) != prefix_bytes:
                raise InjectionError("NEF original truncado durante la lectura del prefijo")

        patched_prefix = bytearray(orig_prefix)
        # 1. Limpiar con ceros todos los huecos de previews originales en el prefijo para no dejar
        #    ningún rastro de la imagen original (incluso si algún preview desborda y se mueve al final).
        wiped_bytes = 0
        for block, preview in zip(plan.previews, ordered_previews):
            if block.original_offset > 0 and block.original_length > 0:
                end_orig = block.original_offset + block.original_length
                if end_orig <= len(patched_prefix):
                    patched_prefix[block.original_offset : end_orig] = b"\x00" * block.original_length
                    wiped_bytes += block.original_length
            if block.in_place:
                if block.offset < 0 or block.offset + block.byte_count > len(patched_prefix):
                    raise InjectionError(f"Preview in-place fuera de prefijo: {block.ifd_path}")
                patched_prefix[block.offset : block.offset + block.byte_count] = preview.data

        # 2. Aplicar parches de punteros y longitudes TIFF / MakerNote
        for patch in plan.patches:
            if patch.location < 0 or patch.location + 4 > len(patched_prefix):
                raise InjectionError("Parche fuera del área de prefijo")
            struct.pack_into(self.ctx.endian + "I", patched_prefix, patch.location, patch.replacement)

        compared = max(0, prefix_bytes - 4 * len(plan.patches) - wiped_bytes)
        padding = b"\0" * plan.preview_padding_bytes
        trailing_previews = [
            preview for block, preview in zip(plan.previews, ordered_previews) if not block.in_place
        ]

        nef_hasher = hashlib.sha256()
        nef_hasher.update(patched_prefix)
        for preview in trailing_previews:
            nef_hasher.update(preview.data)
        nef_hasher.update(padding)
        nef_hasher.update(payload_mv)
        output_nef_sha256 = nef_hasher.hexdigest()

        with provisional_nef.open("wb") as dst:
            dst.write(patched_prefix)
            for preview in trailing_previews:
                dst.write(preview.data)
            dst.write(padding)
            dst.write(payload_mv)
            dst.flush()

        if provisional_nef.stat().st_size != plan.output_size:
            raise InjectionError("Longitud de salida inesperada tras escritura")

        # Reinspeccionar cabeceras TIFF y MakerNote del archivo generado
        parsed = NefInspector(provisional_nef).inspect()
        raw = parsed.get("raw") or {}
        segments = raw.get("segments") or []
        if (
            len(segments) != 1
            or segments[0].get("offset") != plan.payload_offset
            or segments[0].get("byte_count") != plan.payload_bytes
            or not segments[0].get("inside_file")
            or segments[0].get("end_offset_exclusive") != plan.output_size
        ):
            raise InjectionError("El inspector TIFF no resuelve correctamente el nuevo strip")
        previews_by_path = {
            preview.get("ifd_path"): preview for preview in parsed.get("previews", [])
        }
        parsed_mk = NikonMakerNoteInspector(provisional_nef, inspection=parsed).inspect()
        mk_nested = {
            "IFD0/EXIF/MakerNote/NikonPreviewIFD": item
            for item in parsed_mk.get("maker_note", {}).get("nested_ifds", [])
            if item.get("kind") == "NikonPreviewIFD"
        }
        for block in plan.previews:
            if block.ifd_path == "IFD0/EXIF/MakerNote/NikonPreviewIFD":
                mk_prev = mk_nested.get(block.ifd_path)
                jp = mk_prev.get("jpeg_preview") if isinstance(mk_prev, dict) else None
                if (
                    not isinstance(jp, dict)
                    or jp.get("data_offset") != block.offset
                    or jp.get("byte_count") != block.byte_count
                    or not jp.get("inside_makernote")
                    or not jp.get("has_jpeg_soi")
                    or jp.get("width") != block.width
                    or jp.get("height") != block.height
                ):
                    raise InjectionError(
                        f"El inspector MakerNote no resuelve correctamente el preview {block.ifd_path}"
                    )
            else:
                preview = previews_by_path.get(block.ifd_path)
                preview_segments = preview.get("segments") if isinstance(preview, dict) else None
                if (
                    not isinstance(preview_segments, list)
                    or len(preview_segments) != 1
                    or preview.get("width") != block.width
                    or preview.get("height") != block.height
                    or preview_segments[0].get("offset") != block.offset
                    or preview_segments[0].get("byte_count") != block.byte_count
                    or not preview_segments[0].get("inside_file")
                ):
                    raise InjectionError(
                        f"El inspector TIFF no resuelve correctamente el preview {block.ifd_path}"
                    )
        if parsed.get("camera", {}).get("model") != self.ctx.model:
            raise InjectionError("Se ha alterado el modelo de cámara en metadatos")

        container_report = {
            "tiff_pointer_fields_ok": True,
            "raw_payload_sha256": payload_sha256,
            "payload_equal_to_its_bin": True,
            "maker_note_unchanged": not any(
                b.ifd_path == "IFD0/EXIF/MakerNote/NikonPreviewIFD" for b in plan.previews
            ),
            "maker_note_preview_regenerated": any(
                b.ifd_path == "IFD0/EXIF/MakerNote/NikonPreviewIFD" for b in plan.previews
            ),
            "preserved_original_bytes_compared": compared,
            "tiff_reinspection": "pass",
            "embedded_previews_regenerated": bool(plan.previews),
            "embedded_previews": [preview.as_report() for preview in plan.previews],
            "notes": [
                "Los huecos originales de las miniaturas TIFF y MakerNote se limpian con ceros y se sobrescriben in-place cuando caben en su región nativa.",
                (
                    "Las 4 vistas previas (IFD0 RGB8, SubIFD0 JPEG 4:2:2, SubIFD2 JPEG 4:2:2 y MakerNote NikonPreviewIFD JPEG 4:2:2) fueron regeneradas."
                    if plan.previews
                    else "La plantilla no contiene vistas previas TIFF/JPEG regenerables."
                ),
            ],
        }
        return output_nef_sha256, container_report


# ---------------------------------------------------------------------------
# Decodificación y Sonda LibRaw / rawpy
# ---------------------------------------------------------------------------

def find_libraw_cdll(spec: str | None = None) -> ctypes.CDLL | None:
    """Busca y carga libraw nativo (DLL/SO/DYLIB) incluyendo rutas de entornos Conda."""
    candidates: list[str | None] = [
        spec,
        ctypes.util.find_library("raw"),
        "libraw.so.23",
        "libraw.so.22",
        "libraw.so.20",
        "libraw.so",
        "libraw.dll",
        "libraw_r.dll",
        "raw.dll",
        "libraw.dylib",
    ]
    # Añadir posibles rutas de entornos Conda locales en Windows si existen
    for env_path in (
        Path(r"F:\anaconda3\envs\image_3_11\Library\bin\raw.dll"),
        Path(r"F:\anaconda3\envs\imageQA_3_12\Library\bin\raw.dll"),
        Path(r"C:\Users\jose\anaconda3\Library\bin\raw.dll"),
    ):
        if env_path.is_file():
            candidates.append(str(env_path))

    for libname in candidates:
        if not libname:
            continue
        try:
            return ctypes.CDLL(libname)
        except OSError:
            continue
    return None


class LibRawNativeProbe:
    """Verifica la ruta de decodificación mediante la API C pública de LibRaw."""

    def __init__(self, library: str | None = None) -> None:
        self.library = library

    def probe(self, path: str | Path) -> dict[str, Any]:
        lib = find_libraw_cdll(self.library)
        if lib is None:
            raise PayloadDecodeError("No se encontró libraw nativo en este sistema")
        lib.libraw_version.restype = ctypes.c_char_p
        lib.libraw_init.argtypes = [ctypes.c_uint]
        lib.libraw_init.restype = ctypes.c_void_p
        lib.libraw_open_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.libraw_open_file.restype = ctypes.c_int
        lib.libraw_unpack.argtypes = [ctypes.c_void_p]
        lib.libraw_unpack.restype = ctypes.c_int
        lib.libraw_get_raw_width.argtypes = [ctypes.c_void_p]
        lib.libraw_get_raw_width.restype = ctypes.c_int
        lib.libraw_get_raw_height.argtypes = [ctypes.c_void_p]
        lib.libraw_get_raw_height.restype = ctypes.c_int
        lib.libraw_strerror.argtypes = [ctypes.c_int]
        lib.libraw_strerror.restype = ctypes.c_char_p
        lib.libraw_close.argtypes = [ctypes.c_void_p]

        ctx = lib.libraw_init(0)
        if not ctx:
            raise PayloadDecodeError("libraw_init devolvió NULL")
        try:
            err = lib.libraw_open_file(ctx, os.fsencode(Path(path)))
            if err:
                raise PayloadDecodeError(
                    f"libraw_open_file: {lib.libraw_strerror(err).decode('utf-8', 'replace')}"
                )
            raw_width = int(lib.libraw_get_raw_width(ctx))
            raw_height = int(lib.libraw_get_raw_height(ctx))
            err = lib.libraw_unpack(ctx)
            if err:
                raise PayloadDecodeError(
                    f"libraw_unpack: {lib.libraw_strerror(err).decode('utf-8', 'replace')}"
                )
            return {
                "api": "libraw_public_c_api",
                "version": lib.libraw_version().decode("ascii", "replace"),
                "raw_width": raw_width,
                "raw_height": raw_height,
                "unpack_return_code": 0,
                "unpack_success": True,
                "pixel_array_exported": False,
                "note": "Comprueba que el desempaquetador concluye sin error; no comprueba los valores píxel a píxel.",
            }
        finally:
            lib.libraw_close(ctx)


@dataclass(frozen=True)
class CfaGeometry:
    pattern: np.ndarray
    color_desc: str
    top_margin: int
    left_margin: int

    def visible_pattern(self) -> np.ndarray:
        arr = np.asarray(self.pattern)
        p, q = arr.shape
        return arr[
            (np.arange(p)[:, None] + self.top_margin) % p,
            (np.arange(q)[None, :] + self.left_margin) % q,
        ]

    def color_names(self, pattern: np.ndarray | None = None) -> list[list[str]]:
        a = np.asarray(self.pattern if pattern is None else pattern)
        return [[self.color_desc[int(v)] for v in row] for row in a]

    def to_json(self) -> dict[str, Any]:
        v = self.visible_pattern()
        return {
            "coordinate_system": "raw_full_image_origin_top_left",
            "color_desc": self.color_desc,
            "repeat_pattern_dim": list(map(int, self.pattern.shape)),
            "full_pattern_codes": np.asarray(self.pattern, dtype=int).tolist(),
            "full_pattern_names": self.color_names(),
            "visible_pattern_codes": v.astype(int).tolist(),
            "visible_pattern_names": self.color_names(v),
            "visible_phase_offset_modulo_pattern": [
                self.top_margin % self.pattern.shape[0],
                self.left_margin % self.pattern.shape[1],
            ],
        }


@dataclass
class RawFrame:
    image: np.ndarray
    visible_crop: tuple[int, int, int, int]
    cfa: CfaGeometry
    metadata: dict[str, Any]

    @property
    def visible(self) -> np.ndarray:
        t, l, h, w = self.visible_crop
        return self.image[t : t + h, l : l + w]


class RawPyBackend:
    """Envoltorio de rawpy para obtener el mosaico Bayer 2D uint16 sin demosaicing."""

    name = "rawpy_libraw"

    def load(self, path: str | Path) -> RawFrame:
        try:
            import rawpy
        except ImportError as exc:
            raise PayloadDecodeError("rawpy no está instalado.") from exc

        with rawpy.imread(str(path)) as raw:
            if raw.raw_type != rawpy.RawType.Flat:
                raise PayloadDecodeError(f"RAW de tipo no plano: {raw.raw_type}")
            image = np.array(raw.raw_image, dtype=np.uint16, copy=True, order="C")
            pattern = raw.raw_pattern
            sizes = raw.sizes
            desc = raw.color_desc
            if isinstance(desc, bytes):
                desc = desc.decode("ascii", "replace")
            desc = str(desc).rstrip("\x00")
            cfa = CfaGeometry(
                np.array(pattern, dtype=np.int16, copy=True),
                desc,
                int(sizes.top_margin),
                int(sizes.left_margin),
            )
            metadata = {
                "backend_version": str(rawpy.__version__),
                "libraw_version": str(rawpy.libraw_version),
                "raw_type": str(raw.raw_type),
                "raw_height": int(sizes.raw_height),
                "raw_width": int(sizes.raw_width),
                "height": int(sizes.height),
                "width": int(sizes.width),
                "iheight": int(sizes.iheight),
                "iwidth": int(sizes.iwidth),
                "top_margin": int(sizes.top_margin),
                "left_margin": int(sizes.left_margin),
                "flip": int(sizes.flip),
                "num_colors": int(raw.num_colors),
                "black_level_per_channel": [int(x) for x in raw.black_level_per_channel],
                "camera_white_level_per_channel": (
                    None
                    if raw.camera_white_level_per_channel is None
                    else [int(x) for x in raw.camera_white_level_per_channel]
                ),
                "white_level": int(raw.white_level),
                "camera_whitebalance": [float(x) for x in raw.camera_whitebalance],
                "daylight_whitebalance": [float(x) for x in raw.daylight_whitebalance],
                "rgb_xyz_matrix": np.asarray(raw.rgb_xyz_matrix, dtype=float).tolist(),
            }
        return RawFrame(
            image=image,
            visible_crop=(
                int(sizes.top_margin),
                int(sizes.left_margin),
                int(sizes.height),
                int(sizes.width),
            ),
            cfa=cfa,
            metadata=metadata,
        )

