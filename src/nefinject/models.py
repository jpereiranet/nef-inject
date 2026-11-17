"""Modelos de datos inmutables, jerarquía de excepciones y utilidades criptográficas.

Forma parte de la librería `nefinject`, definiendo las estructuras comunes de configuración,
resultados de ejecución, serialización transaccional y validación de integridad SHA-256
para todo el pipeline de síntesis e inyección RAW.

Componentes del módulo:
- Versiones de esquema: `LIBRARY_VERSION`, `SCHEMA_VERSION`, `SUMMARY_SCHEMA_VERSION`
  para garantizar compatibilidad hacia adelante y trazabilidad en los manifiestos JSON.
- Jerarquía de Excepciones: Clase base `NefInjectError` y especializaciones para cada etapa
  del pipeline (`TiffParseError`, `MakerNoteParseError`, `PayloadDecodeError`,
  `CameraColorProbeError`, `DngProfileError`, `ConversionError`, `CodecError`,
  `InjectionError`, `CalibrationError`).
- Utilidades Criptográficas y de E/S Atómica:
  * `sha256_file`: Cálculo determinista del hash SHA-256 de archivos en disco.
  * `sha256_bytes`: Suma SHA-256 de búferes en memoria.
  * `sha256_npy_in_memory`: Hash del flujo binario en formato npy para tensores NumPy.
  * `save_npy_and_sha256`: Guardado atómico en disco junto con su cálculo de huella.
  * `write_json_atomic`: Escritura atómica basada en archivo temporal + renombrado seguro.
  * `read_json` y `to_native`: Lectura y saneamiento recursivo de tipos NumPy a tipos primitivos Python.
- Modelos de Configuración y Resultados:
  * `InjectionConfig`: Dataclass congelada con los parámetros de entrada del pipeline
    (rutas de archivos, balance de blancos, EV, temperatura Kelvin, perfiles de óptica y sensor).
  * `PipelineResult`: Resumen estructurado del experimento con rutas y hashes SHA-256
    de todos los artefactos generados.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LIBRARY_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
SUMMARY_SCHEMA_VERSION = "2.0.0-dng-profile"


# ---------------------------------------------------------------------------
# Jerarquía de Excepciones
# ---------------------------------------------------------------------------

class NefInjectError(RuntimeError):
    """Excepción base de la librería nefinject."""


class TiffParseError(NefInjectError, ValueError):
    """Error estructural del contenedor TIFF clásico."""


class MakerNoteParseError(NefInjectError, ValueError):
    """Datos ausentes, truncados o incompatibles en el bloque MakerNote Nikon."""


class PayloadDecodeError(NefInjectError):
    """Error al desempaquetar o validar el mosaico RAW mediante LibRaw/rawpy."""


class CameraColorProbeError(NefInjectError):
    """Error al extraer metadatos colorimétricos del NEF."""


class DngProfileError(NefInjectError, ValueError):
    """Error al extraer o interpolar el perfil cromático del archivo DNG."""


class ConversionError(NefInjectError, ValueError):
    """Entrada o configuración no válida para la síntesis del mosaico Bayer."""


class CodecError(NefInjectError, ValueError):
    """Contexto incompatible, datos incompletos o error en códec Nikon 14-bit Lossless."""


class InjectionError(NefInjectError):
    """Fallo de prevalidación, inyección transaccional o comprobación del contenedor NEF."""


class CalibrationError(NefInjectError, ValueError):
    """Error de formato, correspondencia de datos o geometría de calibración."""


# ---------------------------------------------------------------------------
# Utilidades SHA-256 (en disco y en memoria sin relectura)
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Calcula el resumen SHA-256 en minúsculas de un archivo en disco."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        while chunk := fd.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes | bytearray | memoryview | np.ndarray) -> str:
    """Calcula el resumen SHA-256 en minúsculas de un buffer en memoria."""
    if isinstance(data, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(data).view(np.uint8)).hexdigest()
    return hashlib.sha256(data).hexdigest()


def sha256_npy_in_memory(array: np.ndarray) -> str:
    """Calcula en memoria exactamente el mismo SHA-256 que tendría un archivo .npy
    escrito con np.save / np.lib.format.open_memmap (formato NPY v1.0),
    evitando releer ~49 MB desde disco.
    """
    arr = np.ascontiguousarray(array)
    header_buf = io.BytesIO()
    header = np.lib.format.header_data_from_array_1_0(arr)
    np.lib.format.write_array_header_1_0(header_buf, header)
    h = hashlib.sha256()
    h.update(header_buf.getvalue())
    h.update(memoryview(arr))
    return h.hexdigest()


def save_npy_and_sha256(path: str | Path, array: np.ndarray, *, overwrite: bool = False) -> str:
    """Escribe un array NumPy en formato .npy de forma atómica y calcula su SHA-256
    en el mismo paso en memoria sin releer el archivo desde disco.
    """
    dest = Path(path)
    if not overwrite and dest.exists():
        raise FileExistsError(f"El archivo ya existe y overwrite=False: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    header_buf = io.BytesIO()
    header = np.lib.format.header_data_from_array_1_0(arr)
    np.lib.format.write_array_header_1_0(header_buf, header)
    header_bytes = header_buf.getvalue()

    h = hashlib.sha256()
    h.update(header_bytes)
    h.update(memoryview(arr))
    digest = h.hexdigest()

    with tempfile.NamedTemporaryFile(
        prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(header_bytes)
            tmp.write(memoryview(arr))
            tmp.flush()
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        if overwrite:
            os.replace(tmp_path, dest)
        else:
            os.link(tmp_path, dest)
            tmp_path.unlink(missing_ok=True)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return digest


# ---------------------------------------------------------------------------
# Serialización JSON Transaccional
# ---------------------------------------------------------------------------

def to_native(obj: Any) -> Any:
    """Convierte recursivamente tipos NumPy y Path a tipos nativos serializables en JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj


def read_json(path: str | Path) -> dict[str, Any]:
    """Lee un archivo JSON y verifica que la raíz sea un diccionario."""
    with Path(path).open("r", encoding="utf-8") as fd:
        data = json.load(fd)
    if not isinstance(data, dict):
        raise NefInjectError(f"Se esperaba un objeto JSON en {path}")
    return data


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    overwrite: bool = False,
) -> str:
    """Escribe un archivo JSON con codificación UTF-8 sin BOM de forma transaccional
    y devuelve su SHA-256 calculado en memoria sin releer de disco.
    """
    dest = Path(path)
    if not overwrite and dest.exists():
        raise FileExistsError(f"El archivo JSON ya existe y overwrite=False: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(to_native(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if os.linesep != "\n":
        serialized = serialized.replace("\n", os.linesep)
    raw_bytes = serialized.encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(raw_bytes)
            tmp.flush()
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        if overwrite:
            os.replace(tmp_path, dest)
        else:
            os.link(tmp_path, dest)
            tmp_path.unlink(missing_ok=True)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return digest


# ---------------------------------------------------------------------------
# Dataclasses de Configuración y Resultados del Pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionConfig:
    """Configuración completa y trazable para la síntesis e inyección de un NEF sintético."""
    nef_path: Path
    dng_path: Path | None = None
    chart_path: Path | None = None
    pattern: str = "colorchecker"          # "colorchecker" | "star"; se usa sin chart_path
    output_dir: Path = Path("experimentos_dng")
    run_dir: Path | None = None
    wb_gains: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    exposure_ev: float = -1.0
    kelvin: float | None = 6200.0
    black_dn: int = 0
    white_dn: int = 15000
    resize_mode: str = "cover"              # "contain" | "cover" | "stretch" | "none"
    orientation: str = "preserve"           # "preserve" | "match-sensor"
    rotation: str | None = None              # None | "cw" | "ccw"
    input_encoding: str = "srgb"            # "srgb" | "linear" | "auto"
    background_linear: float = 0.18
    strategy: str = "replace-tail"          # "replace-tail" | "append"
    expected_profile: str = "Adobe Standard"
    allow_different_original_name: bool = False
    decode_original: bool = False
    save_npz: bool = False
    verification_mode: str = "fast"         # "fast" | "paranoid" | "full" | "none"
    codec_backend: str = "auto"             # "auto" | "numba" | "numpy" | "python"
    persist_intermediates: bool = True      # Si True, escribe los 13 artefactos en disco
    render_validation_png: bool = False     # Si True, genera validacion_rawpy_srgb8.png
    layout_cols: int = 6
    layout_rows: int = 4
    layout_roi_fill: float = 0.5
    layout_neutrals_last_row: bool = True
    custom_layout_path: Path | None = None
    compare_with_dir: Path | None = None
    optics_profile: Any = None              # OpticsConfig | ruta explícita a JSON
    sensor_profile: Any = None              # SensorConfig | ruta explícita a JSON
    save_optics_tiff: bool = False          # Si True, guarda optica_pre_cfa_linear.tif (~294 MB) en disco

    def __post_init__(self) -> None:
        object.__setattr__(self, "nef_path", Path(self.nef_path))
        if self.dng_path is not None:
            object.__setattr__(self, "dng_path", Path(self.dng_path))
        if self.chart_path is not None:
            object.__setattr__(self, "chart_path", Path(self.chart_path))
        if self.pattern not in ("colorchecker", "star"):
            raise ValueError("pattern debe ser colorchecker o star")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.run_dir is not None:
            object.__setattr__(self, "run_dir", Path(self.run_dir))
        if self.custom_layout_path is not None:
            object.__setattr__(self, "custom_layout_path", Path(self.custom_layout_path))
        if self.compare_with_dir is not None:
            object.__setattr__(self, "compare_with_dir", Path(self.compare_with_dir))
        object.__setattr__(self, "wb_gains", tuple(float(g) for g in self.wb_gains))
        if self.orientation not in ("preserve", "match-sensor"):
            raise ValueError("orientation debe ser preserve o match-sensor")
        if self.rotation not in (None, "cw", "ccw"):
            raise ValueError("rotation debe ser cw, ccw o None")
        if self.orientation == "preserve" and self.rotation is not None:
            raise ValueError("rotation requiere orientation='match-sensor'")


@dataclass(frozen=True)
class PipelineResult:
    """Resultado estructurado de una ejecución del pipeline de inyección NEF."""
    run_id: str
    run_dir: Path
    output_nef_path: Path
    output_nef_sha256: str
    bayer_sha256: str
    compressed_payload_sha256: str
    compressed_payload_bytes: int
    rgb_to_camera_matrix: np.ndarray
    summary_report: dict[str, Any]
    injection_report: dict[str, Any]
    bayer_manifest: dict[str, Any]
    dng_profile_report: dict[str, Any] | None
    color_probe_report: dict[str, Any] | None
    inspection_report: dict[str, Any]
    makernote_report: dict[str, Any]
    timings_ms: dict[str, float] = field(default_factory=dict)
    injected_bayer_sha256: str | None = None
    optics_manifest: dict[str, Any] | None = None
    sensor_manifest: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "output_nef_path": str(self.output_nef_path),
            "output_nef_sha256": self.output_nef_sha256,
            "bayer_sha256": self.bayer_sha256,
            "injected_bayer_sha256": self.injected_bayer_sha256 or self.bayer_sha256,
            "compressed_payload_sha256": self.compressed_payload_sha256,
            "compressed_payload_bytes": self.compressed_payload_bytes,
            "rgb_to_camera_matrix": self.rgb_to_camera_matrix.tolist(),
            "optics_enabled": self.optics_manifest is not None,
            "sensor_enabled": self.sensor_manifest is not None,
            "timings_ms": self.timings_ms,
        }
