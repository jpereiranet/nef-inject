"""Módulo de calibración colorimétrica post-inyección, métricas CIEDE2000 y análisis comparativo.

Forma parte de la librería `nefinject`, proporcionando herramientas de validación objetiva,
afinamiento de matrices de color y balance de blancos, y cálculo de diferencias de color
perceptuales normalizadas según los estándares de la CIE (Comisión Internacional de la Iluminación).

Responsabilidades y componentes del módulo:
1. Muestreo de Cartas y Regiones de Interés (`BayerRawSampler`, `ChartMeasurements`, `PatchData`):
   - Extracción de medias y desviaciones típicas en las 24 muestras (parches) de cartas
     Macbeth ColorChecker tanto en mosaicos crudos Bayer (CFA) como en imágenes reveladas (TIFF/sRGB).

2. Calibración y Optimización de Parámetros:
   - `WhiteBalanceCalibrator`: Ajuste iterativo de multiplicadores de balance de blancos
     a partir del parche neutro gris medio (#22) o la escala completa de grises (parches 19 a 24).
   - `ExposureCalibrator`: Ajuste de ganancia lineal (compensación EV) para igualar la luminancia
     de referencia sin introducir saturación en las altas luces.
   - `ColorTransformCalibrator`: Optimización por mínimos cuadrados de la matriz 3x3 de corrección
     de color (RGB de cámara a sRGB/AdobeRGB).
   - `CalibrationRefiner`: Algoritmo de descenso o Nelder-Mead multivariable para optimización
     conjunta de multiplicadores y matriz.

3. Métricas de Fidelidad Perceptual de Color:
   - `srgb_to_linear`, `linear_to_srgb`, `linear_srgb_to_lab`: Conversiones entre espacios
     colorimétricos no lineales sRGB, lineal y CIE 1976 L*a*b* con iluminante de referencia D65.
   - `delta_e_2000`: Implementación completa de la fórmula estándar CIEDE2000 (ISO/CIE 11664-6:2014)
     con factores de ponderación kL, kC, kh y términos de rotación RT para tonos azules.

4. Comparación de Mosaicos y Validación Cruzada (`compare_bayer_syntheses`):
   - Comparación estadística píxel a píxel entre síntesis generadas y referencias experimentales,
     con verificación opcional directa sobre archivos RAW NEF mediante `rawpy`.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .container import TiffReader
from .models import (
    CalibrationError,
    read_json,
    sha256_file,
    to_native,
    write_json_atomic,
)
from .synthesis import make_grid_layout

__version__ = "0.7.0"


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1, EOTF sRGB para valores normalizados en [0,1]."""
    a = np.asarray(rgb, dtype=np.float64)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)


_SRGB_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def linear_srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.einsum("...j,ij->...i", rgb, _SRGB_XYZ) / _D65
    delta = 6.0 / 29.0
    f = np.where(
        xyz > delta**3,
        np.cbrt(np.maximum(xyz, 0.0)),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ),
        axis=-1,
    )


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 (kL=kC=kH=1) según Sharma, Wu & Dalal (2005)."""
    a = np.asarray(lab1, dtype=np.float64)
    b = np.asarray(lab2, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise CalibrationError("Lab de la misma forma (...,3) requerido")
    L1, A1, B1 = np.moveaxis(a, -1, 0)
    L2, A2, B2 = np.moveaxis(b, -1, 0)
    C1 = np.hypot(A1, B1)
    C2 = np.hypot(A2, B2)
    meanC = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - np.sqrt(meanC**7 / (meanC**7 + 25.0**7)))
    a1p = (1.0 + G) * A1
    a2p = (1.0 + G) * A2
    c1p = np.hypot(a1p, B1)
    c2p = np.hypot(a2p, B2)
    h1p = np.mod(np.degrees(np.arctan2(B1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(B2, a2p)), 360.0)
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(c1p * c2p == 0.0, 0.0, dh)
    dL = L2 - L1
    dC = c2p - c1p
    dH = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2.0)
    Lbar = (L1 + L2) / 2.0
    Cbar = (c1p + c2p) / 2.0
    hsum = h1p + h2p
    hbar = np.where(
        c1p * c2p == 0.0,
        hsum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            hsum / 2.0,
            np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
        ),
    )
    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbar))
        + 0.32 * np.cos(np.radians(3.0 * hbar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbar - 63.0))
    )
    sl = 1.0 + (0.015 * (Lbar - 50.0) ** 2) / np.sqrt(20.0 + (Lbar - 50.0) ** 2)
    sc = 1.0 + 0.045 * Cbar
    sh = 1.0 + 0.015 * Cbar * T
    dtheta = 30.0 * np.exp(-(((hbar - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7))
    rt = -rc * np.sin(np.radians(2.0 * dtheta))
    l = dL / sl
    c = dC / sc
    h = dH / sh
    return np.sqrt(np.maximum(l * l + c * c + h * h + rt * c * h, 0.0))


def _y_linear(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb) @ _SRGB_XYZ[1]


def _icc_check(blob: bytes | None, encoding: str, ignore_icc: bool, context: str) -> str:
    if not blob:
        return "absent_assumed_srgb" if encoding == "srgb" else "absent_declared_linear_srgb"
    if ignore_icc:
        return "ignored_by_explicit_request"
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(blob))
        description = ImageCms.getProfileName(prof).strip()
    except Exception as exc:
        raise CalibrationError(f"ICC en {context} no interpretable: {exc}") from exc
    if encoding == "linear":
        raise CalibrationError(
            "TIFF con ICC y --encoding linear: normalizar previamente o usar --ignore-icc de manera explícita"
        )
    if "srgb" not in description.lower() and "s-rgb" not in description.lower():
        raise CalibrationError(
            f"Perfil {description!r} en {context}; exportar TIFF sRGB (o usar --ignore-icc justificadamente)"
        )
    return f"verified_srgb_icc:{description}"


def _bounds(rect: Sequence[float], units: str, w: int, h: int) -> tuple[int, int, int, int]:
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        raise CalibrationError("ROI [left, top, right, bottom] requerido")
    vals = [float(v) for v in rect]
    if not all(math.isfinite(v) for v in vals):
        raise CalibrationError("ROI no finita")
    if units == "relative":
        if not all(0.0 <= v <= 1.0 for v in vals):
            raise CalibrationError(f"ROI relativa fuera [0,1]: {rect}")
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    elif units != "pixels":
        raise CalibrationError(f"Unidades ROI no soportadas: {units}")
    x0, y0, x1, y1 = (
        int(math.floor(vals[0])),
        int(math.floor(vals[1])),
        int(math.ceil(vals[2])),
        int(math.ceil(vals[3])),
    )
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise CalibrationError(f"ROI inválida {rect} en {w}×{h}")
    return x0, y0, x1, y1


def _read_uncompressed_tiff_native(path: Path, encoding: str, ignore_icc: bool) -> tuple[np.ndarray, str]:
    """Lector nativo sin tifffile para archivos TIFF RGB de 8 o 16 bits sin comprimir."""
    reader = TiffReader(path).parse()
    ifd0 = reader.get("IFD0")
    if ifd0 is None:
        raise CalibrationError(f"No se encontró IFD0 en {path}")
    e = reader.endian
    photo = ifd0.num(0x0106, e)
    if photo not in (1, 2):
        raise CalibrationError("TIFF debe ser RGB o monocromático, sin paleta/CMYK")
    orientation = ifd0.num(0x0112, e)
    if orientation is not None and orientation != 1:
        raise CalibrationError("TIFF orientado: normalizar previamente a Orientation=1")
    comp = ifd0.num(0x0103, e) or 1
    if comp != 1:
        raise CalibrationError(
            f"TIFF comprimido (Compression={comp}): instalar tifffile para descomprimir LZW/Deflate"
        )
    icc_tag = ifd0.tag(0x8773)
    icc_bytes = reader.read_tag_bytes(icc_tag) if icc_tag is not None else None
    status = _icc_check(icc_bytes, encoding, ignore_icc, path.name)

    w = ifd0.num(0x0100, e)
    h = ifd0.num(0x0101, e)
    spp = ifd0.num(0x0115, e) or 1
    bps_raw = ifd0.value(0x0102, e)
    bps_list = bps_raw if isinstance(bps_raw, list) else [bps_raw]
    bps = int(bps_list[0])
    if bps not in (8, 16) or not w or not h:
        raise CalibrationError(f"TIFF no soportado por lector nativo: {w}x{h}, {bps} bits")

    strip_offsets = ifd0.value(0x0111, e)
    strip_counts = ifd0.value(0x0117, e)
    offsets = strip_offsets if isinstance(strip_offsets, list) else [strip_offsets]
    counts = strip_counts if isinstance(strip_counts, list) else [strip_counts]

    chunks = []
    with path.open("rb") as fd:
        for off, cnt in zip(offsets, counts):
            fd.seek(int(off))
            chunks.append(fd.read(int(cnt)))
    raw_buf = b"".join(chunks)
    dt = np.dtype(e + ("u2" if bps == 16 else "u1"))
    arr = np.frombuffer(raw_buf, dtype=dt).copy()
    arr = arr.reshape((h, w, spp) if spp > 1 else (h, w))
    return arr, status


@dataclass
class ColorImage:
    array: np.ndarray
    max_value: float
    encoding: str
    profile_status: str
    path: str

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        encoding: str = "srgb",
        ignore_icc: bool = False,
    ) -> "ColorImage":
        path = Path(path)
        if encoding not in ("srgb", "linear"):
            raise CalibrationError("encoding ha de ser srgb o linear")
        if path.suffix.lower() in (".tif", ".tiff"):
            try:
                import tifffile

                with tifffile.TiffFile(path) as tif:
                    page = tif.pages[0]
                    if page.photometric.name.upper() not in ("RGB", "MINISBLACK"):
                        raise CalibrationError("TIFF debe ser RGB o monocromático, sin paleta/CMYK")
                    orientation = page.tags.get(274)
                    if orientation and int(orientation.value) != 1:
                        raise CalibrationError("TIFF orientado: normalizar previamente a Orientation=1")
                    icc_tag = page.tags.get(34675)
                    status = _icc_check(
                        bytes(icc_tag.value) if icc_tag is not None else None,
                        encoding,
                        ignore_icc,
                        path.name,
                    )
                    data = page.asarray()
            except ImportError:
                data, status = _read_uncompressed_tiff_native(path, encoding, ignore_icc)
        else:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                status = _icc_check(im.info.get("icc_profile"), encoding, ignore_icc, path.name)
                if im.mode in ("RGB", "RGBA") and path.suffix.lower() == ".png":
                    with path.open("rb") as fd:
                        hdr = fd.read(26)
                    if hdr[:8] == b"\x89PNG\r\n\x1a\n" and hdr[24] == 16:
                        raise CalibrationError(
                            "PNG RGB 16-bit: convertir a TIFF RGB 16-bit para garantizar precisión"
                        )
                if im.mode not in ("RGB", "RGBA", "L", "I;16", "I"):
                    raise CalibrationError(f"Modo de imagen no admitido: {im.mode}")
                data = np.asarray(im)
        data = np.asarray(data)
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=-1)
        if data.ndim != 3 or data.shape[-1] not in (3, 4):
            raise CalibrationError(f"RGB H×W×3 esperado, recibido {data.shape}")
        data = data[..., :3]
        if data.dtype.kind in "ui":
            scale = float(np.iinfo(data.dtype).max)
        elif data.dtype.kind == "f":
            if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
                raise CalibrationError("TIFF flotante debe estar normalizado a [0,1]")
            scale = 1.0
        else:
            raise CalibrationError(f"Tipo de muestra inválido: {data.dtype}")
        return cls(data, scale, encoding, status, str(path))

    def sample(self, rect: Sequence[float], units: str) -> dict[str, Any]:
        x0, y0, x1, y1 = _bounds(rect, units, self.array.shape[1], self.array.shape[0])
        view = self.array[y0:y1, x0:x1]
        values = view.astype(np.float64) / self.max_value
        clip_fraction = float(np.mean(np.any(values >= 0.998, axis=-1)))
        black_fraction = float(np.mean(np.all(values <= 0.002, axis=-1)))
        if self.encoding == "srgb":
            values = srgb_to_linear(values)
        return {
            "rgb_linear": np.mean(values, axis=(0, 1)).tolist(),
            "rgb_linear_std": np.std(values, axis=(0, 1)).tolist(),
            "clipping_fraction": clip_fraction,
            "near_black_fraction": black_fraction,
            "pixels": int((y1 - y0) * (x1 - x0)),
            "bounds_px": [x0, y0, x1, y1],
        }


@dataclass(frozen=True)
class CameraColorProfile:
    model: str | None
    as_shot_mul_rgbg2: tuple[float, float, float, float]
    candidate_matrix: tuple[tuple[float, float, float], ...]
    probe_sha256: str | None
    source_nef_sha256: str | None
    libraw_version: str | None

    @classmethod
    def from_probe(cls, data: Mapping[str, Any]) -> "CameraColorProfile":
        wb = tuple(float(v) for v in data["white_balance"]["as_shot_cam_mul_rg_bg2"])
        matrix = np.asarray(data["color"]["candidate_linear_srgb_to_camera_3x3"], dtype=np.float64)
        if len(wb) != 4 or not all(math.isfinite(v) and v > 0 for v in wb):
            raise CalibrationError("As-shot WB inválido; cam_mul debe ser positivo para los 4 canales")
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1.0e5:
            raise CalibrationError("Matriz candidata inválida/singular")
        return cls(
            data.get("camera", {}).get("model"),
            wb,  # type: ignore[arg-type]
            tuple(tuple(float(v) for v in row) for row in matrix),
            None,
            data.get("source", {}).get("sha256"),
            data.get("libraw_version"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CameraColorProfile":
        data = read_json(path)
        if data.get("tool", {}).get("name") in ("nefinject.calibration", "nef_color_calibrator"):
            src = data["camera"]
            return cls(
                src.get("model"),
                tuple(src["as_shot_cam_mul_r_g_b_g2"]),
                tuple(tuple(v for v in row) for row in src["candidate_rgb_to_camera_matrix"]),
                src.get("probe_json_sha256"),
                src.get("source_nef_sha256"),
                src.get("libraw_version"),
            )
        c = cls.from_probe(data)
        return cls(
            c.model,
            c.as_shot_mul_rgbg2,
            c.candidate_matrix,
            sha256_file(path),
            c.source_nef_sha256,
            c.libraw_version,
        )

    def wb_inverse_gains(self) -> tuple[float, float, float, float]:
        r, g, b, g2 = self.as_shot_mul_rgbg2
        return (g / r, 1.0, g / g2, g / b)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "as_shot_cam_mul_r_g_b_g2": list(self.as_shot_mul_rgbg2),
            "candidate_rgb_to_camera_matrix": np.asarray(self.candidate_matrix).tolist(),
            "libraw_version": self.libraw_version,
            "probe_json_sha256": self.probe_sha256,
            "source_nef_sha256": self.source_nef_sha256,
        }


@dataclass(frozen=True)
class PatchDefinition:
    id: str
    role: str
    reference_roi: tuple[float, float, float, float]
    rendered_roi: tuple[float, float, float, float]
    raw_roi: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PatchDefinition":
        role = data.get("role", "chromatic")
        if role not in ("neutral", "chromatic"):
            raise CalibrationError(f"Rol de parche inválido {role!r}")
        roi = data.get("reference_roi")
        if roi is None:
            raise CalibrationError("Falta reference_roi")
        return cls(
            str(data["id"]),
            role,
            tuple(roi),
            tuple(data.get("rendered_roi", roi)),
            tuple(data.get("raw_roi", roi)),
        )


class BayerRawSampler:
    """Muestreador CFA real por canales R/G1/G2/B."""

    def __init__(self, path: str | Path, cfa: str, black: float, white: float) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() not in (".npy", ".npz"):
            raise CalibrationError("RAW de entrada .npy o .npz (clave raw)")
        if self.path.suffix.lower() == ".npz":
            with np.load(self.path, allow_pickle=False) as z:
                if "raw" not in z:
                    raise CalibrationError("NPZ debe contener clave raw")
                self.raw = z["raw"].copy()
        else:
            self.raw = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self.raw.ndim != 2 or self.raw.dtype != np.dtype("uint16"):
            raise CalibrationError("Matriz RAW uint16 2D requerida")
        if cfa not in ("RGGB", "BGGR", "GRBG", "GBRG"):
            raise CalibrationError("CFA Bayer de 4 posiciones requerido")
        if not (math.isfinite(black) and math.isfinite(white) and 0 <= black < white):
            raise CalibrationError("Niveles del generador DN deben ser válidos")
        self.cfa = cfa
        self.black = black
        self.white = white
        green_positions = [i for i, ch in enumerate(cfa) if ch == "G"]
        self.positions = {
            cfa.index("R"): "R",
            green_positions[0]: "G1",
            green_positions[1]: "G2",
            cfa.index("B"): "B",
        }

    def sample(self, rect: Sequence[float], units: str) -> dict[str, Any]:
        h, w = self.raw.shape
        x0, y0, x1, y1 = _bounds(rect, units, w, h)
        out: dict[str, float] = {}
        crop = self.raw[y0:y1, x0:x1]
        for i, name in self.positions.items():
            ry, cx = divmod(i, 2)
            iy = (ry - y0) % 2
            ix = (cx - x0) % 2
            data = crop[iy::2, ix::2]
            if not data.size:
                raise CalibrationError("ROI RAW demasiado pequeña para cubrir las cuatro fases CFA")
            normalized = (data.astype(np.float64) - self.black) / (self.white - self.black)
            out[name] = float(np.mean(normalized))
        out["G"] = (out["G1"] + out["G2"]) / 2.0
        return {
            "camera_rgb_linear": [out["R"], out["G"], out["B"]],
            "cfa_linear_r_g1_g2_b": [out["R"], out["G1"], out["G2"], out["B"]],
            "raw_roi_px": [x0, y0, x1, y1],
            "raw_normalization": {
                "black_dn": self.black,
                "white_dn": self.white,
                "source": "bayer synthesis generator configuration",
            },
            "near_white": bool(np.any(crop >= self.white)),
            "near_black": bool(np.any(crop <= self.black)),
        }


@dataclass
class ChartMeasurements:
    patches: list[dict[str, Any]]
    source: dict[str, Any]
    profile: CameraColorProfile | None

    @classmethod
    def measure(
        cls,
        reference_path: str | Path,
        rendered_path: str | Path,
        layout: dict[str, Any],
        *,
        raw_path: str | Path | None = None,
        manifest: dict[str, Any] | None = None,
        profile: CameraColorProfile | None = None,
        encoding: str = "srgb",
        ignore_icc: bool = False,
    ) -> "ChartMeasurements":
        units = layout.get("roi_units", "relative")
        defs = [PatchDefinition.from_mapping(o) for o in layout["patches"]]
        if len({p.id for p in defs}) != len(defs) or not defs:
            raise CalibrationError("Parches vacíos o identificadores duplicados")
        ref = ColorImage.open(reference_path, encoding=encoding, ignore_icc=ignore_icc)
        rnd = ColorImage.open(rendered_path, encoding=encoding, ignore_icc=ignore_icc)
        sampler = None
        if raw_path:
            if not manifest:
                raise CalibrationError("Para analizar .npy/.npz RAW es obligatorio --manifest")
            proc = manifest["processing"]
            target = manifest["target"]
            sampler = BayerRawSampler(
                raw_path, str(target["cfa"]), float(proc["black_dn"]), float(proc["white_dn"])
            )
            if (sampler.raw.shape[1], sampler.raw.shape[0]) != (
                int(target["width"]),
                int(target["height"]),
            ):
                raise CalibrationError("RAW incompatible con las dimensiones del manifiesto")
            if (
                "filename" in manifest.get("array", {})
                and manifest["array"].get("sha256")
                and Path(raw_path).suffix == ".npy"
            ):
                if sha256_file(raw_path) != manifest["array"]["sha256"]:
                    raise CalibrationError("SHA-256 del RAW difiere del manifiesto")
        patches = []
        for p in defs:
            a = ref.sample(p.reference_roi, units)
            b = rnd.sample(p.rendered_roi, units)
            c = sampler.sample(p.raw_roi, units) if sampler else None
            lab1 = linear_srgb_to_lab(np.array(a["rgb_linear"]))
            lab2 = linear_srgb_to_lab(np.array(b["rgb_linear"]))
            patches.append(
                {
                    "id": p.id,
                    "role": p.role,
                    "reference": a,
                    "rendered": b,
                    "raw": c,
                    "color": {
                        "reference_lab": lab1.tolist(),
                        "rendered_lab": lab2.tolist(),
                        "delta_e_00": float(delta_e_2000(lab1, lab2)),
                        "delta_e_76": float(np.linalg.norm(lab1 - lab2)),
                        "reference_y": float(_y_linear(a["rgb_linear"])),
                        "rendered_y": float(_y_linear(b["rgb_linear"])),
                        "rendered_chroma_cab": float(np.hypot(lab2[1], lab2[2])),
                    },
                }
            )
        source = {
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
                "shape_hw": list(ref.array.shape[:2]),
                "encoding": encoding,
                "icc_check": ref.profile_status,
            },
            "rendered": {
                "path": str(rendered_path),
                "sha256": sha256_file(rendered_path),
                "shape_hw": list(rnd.array.shape[:2]),
                "encoding": encoding,
                "icc_check": rnd.profile_status,
            },
            "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path)} if raw_path else None,
            "roi_units": units,
            "layout_patches": len(defs),
        }
        return cls(patches, source, profile)


class WhiteBalanceCalibrator:
    def analyze(self, measurements: ChartMeasurements, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        neutral = [p for p in measurements.patches if p["role"] == "neutral"]
        if not neutral:
            return {"status": "no_neutral_patches", "warning": "Marcar role=neutral en el layout"}
        rendered = []
        raw_ratios = []
        p = measurements.profile
        for pp in neutral:
            v = np.asarray(pp["rendered"]["rgb_linear"])
            if np.min(v) < 0.015 or pp["rendered"]["clipping_fraction"] > 0.002:
                continue
            rendered.append((v / v[1]).tolist())
            if pp["raw"] and p:
                s = np.asarray(pp["raw"]["cfa_linear_r_g1_g2_b"], dtype=float)
                wb = np.asarray(
                    [
                        p.as_shot_mul_rgbg2[0],
                        p.as_shot_mul_rgbg2[1],
                        p.as_shot_mul_rgbg2[3],
                        p.as_shot_mul_rgbg2[2],
                    ],
                    dtype=float,
                )
                if np.any(s <= 0.001) or pp["raw"]["near_white"]:
                    continue
                balanced = s * wb
                green = (balanced[1] + balanced[2]) / 2.0
                raw_ratios.append([float(green / balanced[0]), 1.0, 1.0, float(green / balanced[3])])
        result: dict[str, Any] = {
            "status": "ok" if rendered else "insufficient_valid_neutrals",
            "neutral_patches_total": len(neutral),
            "valid_rendered_neutrals": len(rendered),
            "median_rendered_relative_rgb_r_g_b": np.median(rendered, axis=0).tolist() if rendered else None,
            "mean_rendered_chroma_cab": float(np.mean([q["color"]["rendered_chroma_cab"] for q in neutral])),
            "relative_raw_gains_r_g1_g2_b": None,
            "recommended_generator_gains_r_g1_g2_b": None,
            "raw_balance_status": "no_raw_or_probe",
            "caution": "Kelvin/tint del revelador no se estima de ratios RGB; WB en TIFF y en RAW se informan por separado.",
        }
        if raw_ratios:
            a = np.asarray(raw_ratios)
            r = np.median(a, axis=0)
            result["relative_raw_gains_r_g1_g2_b"] = r.tolist()
            result["raw_balance_status"] = "estimated_from_as_shot_multiplier_and_cfa_neutrals"
            result["valid_raw_neutrals"] = len(a)
            result["relative_raw_gains_mad"] = np.median(np.abs(a - r), axis=0).tolist()
            if manifest:
                original = np.asarray(manifest["processing"]["gains_r_g1_g2_b"], dtype=float)
                result["recommended_generator_gains_r_g1_g2_b"] = (original * r).tolist()
        return result


class ExposureCalibrator:
    def analyze(self, measurements: ChartMeasurements) -> dict[str, Any]:
        neutral = [p for p in measurements.patches if p["role"] == "neutral"]
        used = []
        for p in neutral:
            yr = p["color"]["reference_y"]
            ym = p["color"]["rendered_y"]
            if 0.025 <= yr <= 0.85 and 0.025 <= ym <= 0.85 and p["rendered"]["clipping_fraction"] < 0.002:
                used.append((p["id"], yr, ym))
        if len(used) < 3:
            return {
                "status": "insufficient_valid_neutrals",
                "used_patch_ids": [p[0] for p in used],
                "estimated_ev": None,
                "warning": "Se necesitan >=3 grises no saturados en intervalo lineal de trabajo.",
            }
        ref = np.array([p[1] for p in used])
        meas = np.array([p[2] for p in used])
        if float(np.ptp(np.log2(ref))) < 0.5:
            return {
                "status": "insufficient_luminance_variation",
                "used_patch_ids": [p[0] for p in used],
                "estimated_ev": None,
                "warning": "Se necesitan grises con al menos 0.5 EV de separación en referencia para estimar pendiente tonal.",
            }
        ev = np.log2(ref / meas)
        median_ev = float(np.median(ev))
        mad_ev = float(np.median(np.abs(ev - median_ev)))
        mat = np.column_stack((np.ones(len(ref)), np.log2(ref)))
        coef = np.linalg.lstsq(mat, np.log2(meas), rcond=None)[0]
        predicted = mat @ coef
        res = np.log2(meas) - predicted
        ss_res = float(np.sum(res**2))
        ss_tot = float(np.sum((np.log2(meas) - np.log2(meas).mean()) ** 2))
        gamma = float(coef[1])
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else None
        advisable = abs(gamma - 1.0) < 0.08 and mad_ev < 0.12
        return {
            "status": "approximately_linear" if advisable else "nonlinear_or_unstable",
            "used_patch_ids": [p[0] for p in used],
            "neutral_reference_y": [float(v) for v in ref],
            "neutral_rendered_y": [float(v) for v in meas],
            "estimated_ev": median_ev,
            "ev_per_patch": [float(v) for v in ev],
            "ev_mad": mad_ev,
            "tone_power_law": {"alpha": float(2.0 ** coef[0]), "gamma": gamma, "r_squared": r2},
            "input_exposure_ev_candidate": median_ev if advisable else None,
            "note": "EV estimado como log2(Yref/Ytiff) en sRGB lineal; confirmar mediante barrido de exposición del inyector.",
        }


class ColorTransformCalibrator:
    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        if len(x) < 6 or np.linalg.matrix_rank(x) < 3:
            return {"status": "insufficient_rank_or_patches"}
        cond = float(np.linalg.cond(x))
        if not np.isfinite(cond) or cond > 1.0e5:
            return {"status": "ill_conditioned", "condition_number": cond}
        B = np.linalg.lstsq(x, y, rcond=None)[0]
        yhat = x @ B
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        loo = []
        for i in range(len(x)):
            subset = np.arange(len(x)) != i
            if np.linalg.matrix_rank(x[subset]) < 3:
                continue
            Bi = np.linalg.lstsq(x[subset], y[subset], rcond=None)[0]
            loo.append(float(np.sqrt(np.mean((x[i] @ Bi - y[i]) ** 2))))
        forward = B.T
        invertible = np.linalg.matrix_rank(forward) == 3 and np.linalg.cond(forward) < 1.0e5
        inverse = np.linalg.inv(forward).tolist() if invertible else None
        return {
            "status": "ok",
            "camera_to_output_or_reference_to_output_3x3": forward.tolist(),
            "inverse_3x3": inverse,
            "condition_number": cond,
            "rmse_linear_rgb": rmse,
            "loo_rmse_linear_rgb": float(np.mean(loo)) if loo else None,
            "n_patches": len(x),
            "caution": "Ajuste empírico de TIFF (incluye perfil, WB, tono y posible clipping); no es una matriz de color física.",
        }

    def analyze(self, measurements: ChartMeasurements) -> dict[str, Any]:
        eligible = [
            p
            for p in measurements.patches
            if p["rendered"]["clipping_fraction"] < 0.002
            and p["reference"]["clipping_fraction"] < 0.002
            and 0.02 < p["color"]["reference_y"] < 0.9
            and 0.02 < p["color"]["rendered_y"] < 0.9
        ]
        if len(eligible) < 6:
            return {
                "status": "insufficient_valid_patches",
                "n_valid": len(eligible),
                "note": "Se precisan >=6 parches cromáticamente independientes sin clipping.",
            }
        ref = np.array([p["reference"]["rgb_linear"] for p in eligible])
        out = np.array([p["rendered"]["rgb_linear"] for p in eligible])
        ans: dict[str, Any] = {
            "status": "evaluated",
            "used_patch_ids": [p["id"] for p in eligible],
            "reference_linear_rgb_to_output_linear_rgb": self._fit(ref, out),
            "cfa_camera_linear_to_output_linear_rgb": None,
            "scientific_limit": "No ajustar matrices por revelador antes de comparar su fidelidad cromática; sería evaluación circular.",
        }
        rr = [p for p in eligible if p["raw"] is not None]
        if len(rr) >= 6:
            raw = np.array([p["raw"]["camera_rgb_linear"] for p in rr])
            rendered = np.array([p["rendered"]["rgb_linear"] for p in rr])
            ans["cfa_camera_linear_to_output_linear_rgb"] = self._fit(raw, rendered)
        return ans


class NefColorCalibrator:
    """Analizador cuantitativo ΔE00, balance de blancos y exposición sobre TIFF revelados."""

    def __init__(self, profile: CameraColorProfile | None = None) -> None:
        self.profile = profile

    def analyze(
        self,
        reference: str | Path,
        rendered: str | Path,
        layout: dict[str, Any],
        *,
        raw_path: str | Path | None = None,
        manifest: dict[str, Any] | None = None,
        encoding: str = "srgb",
        ignore_icc: bool = False,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            self.profile
            and manifest
            and self.profile.source_nef_sha256
            and manifest.get("target", {}).get("reference_nef_sha256")
        ):
            if self.profile.source_nef_sha256 != manifest["target"]["reference_nef_sha256"]:
                raise CalibrationError("Perfil de cámara y manifiesto RAW no corresponden al mismo NEF original")
        ms = ChartMeasurements.measure(
            reference,
            rendered,
            layout,
            raw_path=raw_path,
            manifest=manifest,
            profile=self.profile,
            encoding=encoding,
            ignore_icc=ignore_icc,
        )
        wb = WhiteBalanceCalibrator().analyze(ms, manifest)
        exposure = ExposureCalibrator().analyze(ms)
        color = ColorTransformCalibrator().analyze(ms)
        d00 = np.array([p["color"]["delta_e_00"] for p in ms.patches])
        neutrals = np.array([p["color"]["delta_e_00"] for p in ms.patches if p["role"] == "neutral"])
        warnings = [
            "El perfil del revelador y su configuración deben permanecer fijos y documentados.",
            "ΔE00 refiere exclusivamente a la diferencia entre dos imágenes digitales sRGB (no demuestra fidelidad de la cámara real).",
            "La coincidencia cromática mediante matriz empírica ajustada por revelador no permite compararlos imparcialmente.",
            "Comprobar manualmente ROI frente a desplazamiento de área activa y recorte del revelador.",
        ]
        if self.profile is None:
            warnings.append("Sin probe de cámara: no se calcula corrección As Shot en dominio CFA.")
        if "ignored_by_explicit_request" in (
            ms.source["reference"]["icc_check"],
            ms.source["rendered"]["icc_check"],
        ):
            warnings.append("Perfil ICC ignorado explícitamente.")
        if ms.source["reference"]["icc_check"].startswith("absent") or ms.source["rendered"][
            "icc_check"
        ].startswith("absent"):
            warnings.append("Al menos una imagen carece de perfil ICC; se ha SUPUESTO espacio sRGB.")
        if wb.get("relative_raw_gains_r_g1_g2_b") is not None:
            warnings.append("WB RAW se estima con multiplicadores LibRaw.")
        return {
            "schema_version": "1.0.0",
            "tool": {"name": "nefinject.calibration", "version": __version__, "read_only": True},
            "analysis_mode": "digital_srgb_comparison",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": ms.source,
            "camera": self.profile.as_dict() if self.profile else None,
            "render_context": context or {},
            "measurements": ms.patches,
            "summary": {
                "patch_count": len(ms.patches),
                "mean_delta_e00": float(np.mean(d00)),
                "median_delta_e00": float(np.median(d00)),
                "max_delta_e00": float(np.max(d00)),
                "neutral_mean_delta_e00": float(np.mean(neutrals)) if neutrals.size else None,
            },
            "white_balance": wb,
            "exposure": exposure,
            "color_transform": color,
            "warnings": warnings,
        }


class CalibrationRefiner:
    """Refina ganancias CFA en el dominio RAW a partir de los parches neutros analizados."""

    def refine(
        self, report: dict[str, Any], manifest: dict[str, Any], *, apply_exposure: bool = False
    ) -> dict[str, Any]:
        wb = report.get("white_balance", {})
        relative = wb.get("relative_raw_gains_r_g1_g2_b")
        if (
            relative is None
            or wb.get("raw_balance_status") != "estimated_from_as_shot_multiplier_and_cfa_neutrals"
        ):
            raise CalibrationError(
                "Calibración RAW As Shot no disponible; analizar con --raw-npy, --manifest y --profile"
            )
        raw_src = report.get("source", {}).get("raw") or {}
        expected = manifest.get("array", {}).get("sha256")
        if not raw_src.get("sha256") or not expected or raw_src["sha256"] != expected:
            raise CalibrationError("El informe no corresponde a la matriz RAW descrita por el manifiesto")
        old = manifest["processing"]
        gains = np.asarray(old["gains_r_g1_g2_b"], dtype=np.float64)
        ratios = np.asarray(relative, dtype=np.float64)
        proposed = gains * ratios
        if proposed.shape != (4,) or not np.isfinite(proposed).all() or np.any(proposed <= 0):
            raise CalibrationError("Ganancias RAW corregidas inválidas")
        exposure_ev = float(old["exposure_ev"])
        exposure_meta: dict[str, Any] = {
            "applied": False,
            "candidate_ev_from_rendered": report.get("exposure", {}).get("input_exposure_ev_candidate"),
        }
        if apply_exposure:
            inc = report.get("exposure", {}).get("input_exposure_ev_candidate")
            if inc is None or not math.isfinite(inc) or abs(inc) > 2.5:
                raise CalibrationError("No hay una compensación EV plausible")
            exposure_ev += inc
            exposure_meta = {
                "applied": True,
                "change_ev": float(inc),
                "caution": "Aproximación por curva tonal del TIFF: validar experimentalmente la nueva entrada RAW.",
            }
        return {
            "mode": "refined_from_as_shot_raw_neutrals",
            "rgb_to_camera_matrix": old["rgb_to_camera_matrix"],
            "gains_r_g1_g2_b": proposed.tolist(),
            "black_dn": old["black_dn"],
            "white_dn": old["white_dn"],
            "exposure_ev": exposure_ev,
            "exposure_adjustment": exposure_meta,
            "baseline_gains_r_g1_g2_b": gains.tolist(),
            "relative_raw_gain_correction_r_g1_g2_b": ratios.tolist(),
            "black_white_provenance": "generator_manifest_not_measured",
            "notes": [
                "No se modifica el NEF; regenerar la matriz CFA y reinyectarla.",
                "No se han aplicado matrices empíricas de inversión del revelador.",
                "El objetivo es neutralidad As Shot aproximada de LibRaw, no fidelidad espectral Nikon.",
            ],
        }


def compare_bayer_syntheses(
    base_npy: str | Path,
    test_npy: str | Path,
    *,
    nef_base: str | Path | None = None,
    nef_test: str | Path | None = None,
) -> dict[str, Any]:
    """Compara dos mosaicos Bayer .npy (y opcionalmente sus NEF inyectados) devolviendo métricas detalladas."""
    p_base, p_test = Path(base_npy), Path(test_npy)
    a = np.load(p_base, mmap_mode="r", allow_pickle=False)
    b = np.load(p_test, mmap_mode="r", allow_pickle=False)
    if a.shape != b.shape or a.dtype != b.dtype or len(a.shape) != 2:
        raise ValueError("Los dos archivos deben ser mosaicos uint16 Bayer de igual forma")
    ma = read_json(p_base.with_suffix(".json"))
    mb = read_json(p_test.with_suffix(".json"))
    pa, pb = ma["processing"], mb["processing"]
    ca = ma["target"]["cfa"]

    diff_count = int(np.count_nonzero(a != b))
    black = float(pa["black_dn"])
    span = float(pa["white_dn"] - pa["black_dn"])
    delta_ev = pb["exposure_ev"] - pa["exposure_ev"]
    gain_a = np.array(pa["gains_r_g1_g2_b"], dtype=float)
    gain_b = np.array(pb["gains_r_g1_g2_b"], dtype=float)
    expected = np.exp2(delta_ev) * gain_b / gain_a
    names = {"R": 0, "G1": 1, "G2": 2, "B": 3}
    rng = np.random.default_rng(1611)
    green_seen = 0
    channel_ratios: dict[str, Any] = {}
    for pos, color in enumerate(ca):
        row, col = divmod(pos, 2)
        name = color
        if color == "G":
            green_seen += 1
            name = f"G{green_seen}"
        cha, chb = a[row::2, col::2], b[row::2, col::2]
        nrow, ncol = cha.shape
        take = min(nrow * ncol, 100_000)
        ri = rng.integers(0, nrow, size=take)
        ci = rng.integers(0, ncol, size=take)
        va = cha[ri, ci].astype(np.float64) - black
        vb = chb[ri, ci].astype(np.float64) - black
        ok = (va > 0.02 * span) & (cha[ri, ci] < pa["white_dn"] - 5) & (chb[ri, ci] < pb["white_dn"] - 5)
        est = float(np.median(vb[ok] / va[ok])) if np.any(ok) else float("nan")
        idx = names[name]
        channel_ratios[name] = {
            "observed_ratio": est,
            "predicted_ratio": float(expected[idx]),
            "valid_samples": int(ok.sum()),
        }

    nef_checks: dict[str, Any] = {}
    if nef_base or nef_test:
        import rawpy

        for label, nef_p, exp_arr in (("base", nef_base, a), ("test", nef_test, b)):
            if nef_p is not None:
                with rawpy.imread(str(nef_p)) as raw:
                    mism = int(np.count_nonzero(raw.raw_image != exp_arr))
                    nef_checks[label] = {"mismatched_samples": mism, "status": "pass" if mism == 0 else "fail"}
                    if mism:
                        raise RuntimeError(f"{nef_p} difiere en {mism} fotositos respecto a su NPY")

    return {
        "mismatched_photosites": diff_count,
        "total_photosites": int(a.size),
        "mismatch_percent": 100.0 * diff_count / a.size,
        "matrix_equal": bool(np.allclose(pa["rgb_to_camera_matrix"], pb["rgb_to_camera_matrix"], rtol=0, atol=1e-8)),
        "channel_ratios": channel_ratios,
        "nef_checks": nef_checks,
    }

