"""Módulo de síntesis de mosaicos CFA Bayer de 14 bits y generación de patrones de prueba.

Forma parte de la librería `nefinject`, proporcionando algoritmos para la conversión de
imágenes fuente (sRGB / lineal) a mosaicos crudos CFA (Color Filter Array) con patrón Bayer
(RGGB, BGGR, GRBG, GBRG), simulando la respuesta de sensels del sensor fotográfico.

Responsabilidades y componentes del módulo:
1. Conversión de Imagen a Mosaico Bayer (`ImageToBayerConverter`, `BayerConversionConfig`):
   - Decodificación y linealización electro-óptica sRGB según estándar IEC 61966-2-1 (`_eotf_srgb`).
   - Aceleración mediante tabla de correspondencia LUT precalculada de 256 niveles (`_SRGB_U8_LUT_F32`).
   - Gestión precisa de márgenes ópticos y recorte de área activa (`_layout`).
   - Transformación de color del espacio de entrada a espacio de cámara mediante matriz 3x3 (`matrix`).
   - Escalamiento de exposición (compensación EV), aplicación de ganancias por canal de color
     (R, G1, G2, B) y fijación (clamping) a niveles de negro y saturación del sensor (DN de 14 bits).

2. Generación Sintética de Cartas de Calibración (`synthetic_macbeth_image`, `macbeth_colorchecker_rois`):
   - Creación procedimental de la carta Macbeth ColorChecker clásica de 24 parches con
     coordenadas sRGB estandarizadas y definición analítica de regiones de interés (ROIs).

3. Generador de Patrones de Prueba de Frecuencia y Mosaico (`BayerPatternGenerator`, `BayerPatternConfig`):
   - Síntesis analítica de zonas de Fresnel, barridos de frecuencia espacial sinusoidal y patrones
     de impulsos para evaluación de resolución de Nyquist, muaré óptico y artefactos de desmosaizado.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .models import (
    ConversionError,
    save_npy_and_sha256,
    sha256_file,
    sha256_npy_in_memory,
    write_json_atomic,
)

VERSION = "0.6.0"
_VALID_CFA = {"RGGB", "GRBG", "GBRG", "BGGR"}


def _eotf_srgb(c: np.ndarray) -> np.ndarray:
    """EOTF IEC 61966-2-1 (deserialización sRGB a espacio lineal)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oetf_srgb(c: np.ndarray) -> np.ndarray:
    """OETF IEC 61966-2-1 para codificar RGB lineal como sRGB."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


# Tabla LUT precalculada de 256 niveles para imágenes uint8 sRGB (aceleración 10x bit-exacta)
_SRGB_U8_NORM_F32 = np.arange(256, dtype=np.float32) / np.float32(255.0)
_SRGB_U8_LUT_F32 = _eotf_srgb(_SRGB_U8_NORM_F32).astype(np.float32)


@dataclass(frozen=True)
class BayerTarget:
    width: int
    height: int
    cfa: str = "RGGB"
    bit_depth: int = 14
    reference_sha256: str | None = None
    source_model: str | None = None

    @classmethod
    def from_inspection(cls, inspection: str | Path | dict[str, Any]) -> "BayerTarget":
        data = (
            inspection
            if isinstance(inspection, dict)
            else json.loads(Path(inspection).read_text(encoding="utf-8"))
        )
        raw = data["raw"]
        return cls(
            width=int(raw["width"]),
            height=int(raw["height"]),
            cfa=str(raw["cfa"]["pattern"]).upper(),
            bit_depth=int(raw["bits_per_sample"][0]),
            reference_sha256=data.get("source", {}).get("sha256"),
            source_model=data.get("camera", {}).get("model"),
        )

    def validate(self) -> None:
        if self.width < 2 or self.height < 2 or self.width % 2 or self.height % 2:
            raise ConversionError("Para CFA Bayer se requieren dimensiones pares y >= 2")
        if self.cfa not in _VALID_CFA:
            raise ConversionError(f"CFA no admitido: {self.cfa}")
        if not (8 <= self.bit_depth <= 16):
            raise ConversionError("La profundidad ha de estar entre 8 y 16 bits")


@dataclass(frozen=True)
class BayerConversionConfig:
    resize: str = "cover"             # contain | cover | stretch | none
    orientation: str = "preserve"     # preserve | match-sensor
    rotation: str | None = None        # None | cw | ccw
    input_encoding: str = "auto"      # auto | srgb | linear
    black_dn: int = 512
    white_dn: int = 15000
    background_linear: float = 0.18
    exposure_ev: float = 0.0
    gains: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    rgb_to_camera: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    ignore_icc: bool = False

    def validate(self, target: BayerTarget) -> None:
        target.validate()
        if self.resize not in ("contain", "cover", "stretch", "none"):
            raise ConversionError("resize debe ser contain, cover, stretch o none")
        if self.orientation not in ("preserve", "match-sensor"):
            raise ConversionError("orientation debe ser preserve o match-sensor")
        if self.rotation not in (None, "cw", "ccw"):
            raise ConversionError("rotation debe ser cw, ccw o None")
        if self.orientation == "preserve" and self.rotation is not None:
            raise ConversionError("rotation requiere orientation='match-sensor'")
        if self.input_encoding not in ("auto", "srgb", "linear"):
            raise ConversionError("input_encoding debe ser auto, srgb o linear")
        max_dn = (1 << target.bit_depth) - 1
        if not (0 <= self.black_dn < self.white_dn <= max_dn):
            raise ConversionError(f"Intervalo DN inválido para {target.bit_depth} bits [0,{max_dn}]")
        if not (0 <= self.background_linear <= 1):
            raise ConversionError("background_linear fuera de [0,1]")
        if not math.isfinite(self.exposure_ev) or not -12 <= self.exposure_ev <= 12:
            raise ConversionError("exposure_ev fuera de [-12,12]")
        if len(self.gains) != 4 or not all(math.isfinite(x) and x > 0 for x in self.gains):
            raise ConversionError("gains debe tener 4 valores positivos finitos")
        a = np.asarray(self.rgb_to_camera, dtype=float)
        if a.shape != (3, 3) or not np.isfinite(a).all():
            raise ConversionError("rgb_to_camera debe ser una matriz 3x3 finita")


@dataclass
class LoadedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    source_dtype: str
    source_max: float | None
    backend: str
    icc_handling: str
    notes: list[str]


def _orientation_name(height: int, width: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _orient_loaded_image(
    loaded: LoadedImage,
    *,
    target_width: int,
    target_height: int,
    orientation: str,
    rotation: str | None,
) -> tuple[LoadedImage, dict[str, Any]]:
    """Rota la entrada cuando se solicita; nunca altera el mosaico RAW destino."""
    if orientation not in ("preserve", "match-sensor"):
        raise ConversionError("orientation debe ser preserve o match-sensor")
    if rotation not in (None, "cw", "ccw"):
        raise ConversionError("rotation debe ser cw, ccw o None")
    if orientation == "preserve" and rotation is not None:
        raise ConversionError("rotation requiere orientation='match-sensor'")

    source_height, source_width = loaded.rgb.shape[:2]
    source_orientation = _orientation_name(source_height, source_width)
    target_orientation = _orientation_name(target_height, target_width)
    details: dict[str, Any] = {
        "mode": orientation,
        "requested_rotation": rotation,
        "source_shape_hw": [source_height, source_width],
        "target_shape_hw": [target_height, target_width],
        "source_orientation": source_orientation,
        "target_orientation": target_orientation,
        "applied_rotation": "none",
        "degrees": 0,
        "effective_shape_hw": [source_height, source_width],
    }
    needs_rotation = (
        orientation == "match-sensor"
        and source_orientation != "square"
        and target_orientation != "square"
        and source_orientation != target_orientation
    )
    if not needs_rotation:
        return loaded, details
    if rotation is None:
        raise ConversionError(
            "orientation='match-sensor' requiere rotation='cw' o 'ccw' cuando la carta y el sensor tienen orientaciones distintas"
        )

    turns = -1 if rotation == "cw" else 1
    rgb = np.ascontiguousarray(np.rot90(loaded.rgb, k=turns))
    alpha = None if loaded.alpha is None else np.ascontiguousarray(np.rot90(loaded.alpha, k=turns))
    effective_height, effective_width = rgb.shape[:2]
    details.update(
        {
            "applied_rotation": rotation,
            "degrees": 90 if rotation == "cw" else -90,
            "effective_shape_hw": [effective_height, effective_width],
        }
    )
    return LoadedImage(
        rgb,
        alpha,
        loaded.source_dtype,
        loaded.source_max,
        loaded.backend,
        loaded.icc_handling,
        [*loaded.notes, f"Imagen rotada {rotation} para igualar la orientación del sensor"],
    ), details


def _image_channels(array: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2), None
    if array.ndim != 3 or array.shape[-1] not in (1, 2, 3, 4):
        raise ConversionError(f"Canales no compatibles: {array.shape}")
    n = array.shape[-1]
    if n == 1:
        return np.repeat(array, 3, axis=2), None
    if n == 2:
        return np.repeat(array[:, :, :1], 3, axis=2), array[:, :, 1]
    return array[:, :, :3], array[:, :, 3] if n == 4 else None


def _read_png_bit_depth(path: Path) -> int | None:
    with path.open("rb") as fd:
        header = fd.read(26)
    if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        return header[24]
    return None


def _read_with_pillow(path: Path, ignore_icc: bool) -> LoadedImage:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        if im.mode in ("I;16", "I;16B", "I;16L", "I") and not icc:
            a = np.asarray(im)
            if a.dtype == np.int32:
                if a.min() < 0 or a.max() > 65535:
                    raise ConversionError("Imagen I de 32 bits fuera del rango 0-65535")
                a = a.astype(np.uint16)
            rgb, alpha = _image_channels(a)
            return LoadedImage(
                rgb, alpha, str(a.dtype), float(np.iinfo(a.dtype).max), "pillow", "absent", []
            )
        if im.mode == "P" and "transparency" in im.info:
            rgba = im.convert("RGBA")
            alpha_plane = np.asarray(rgba.getchannel("A"))
        else:
            alpha_plane = np.asarray(im.getchannel("A")) if "A" in im.getbands() else None
        notes: list[str] = []
        icc_handling = "absent"
        if icc and not ignore_icc:
            try:
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_profile = ImageCms.createProfile("sRGB")
                rgb_im = im.convert("RGB") if im.mode in ("RGBA", "LA", "P") else im
                rgb_im = ImageCms.profileToProfile(rgb_im, src_profile, dst_profile, outputMode="RGB")
                icc_handling = "converted_to_srgb"
            except Exception as exc:
                raise ConversionError(
                    f"Perfil ICC no aplicable ({exc}); utilizar --ignore-icc solo si se conoce el espacio"
                ) from exc
        else:
            rgb_im = im.convert("RGB")
            if icc:
                icc_handling = "ignored_by_request"
                notes.append("ICC ignorado explícitamente: se supone codificación RGB compatible con sRGB")
            else:
                notes.append("Imagen sin perfil ICC: se supone codificación sRGB para imágenes de 8 bits")
        rgb = np.asarray(rgb_im).copy()
        return LoadedImage(rgb, alpha_plane, str(rgb.dtype), 255.0, "pillow", icc_handling, notes)


def load_image(path: str | Path, *, ignore_icc: bool = False) -> LoadedImage:
    """Lee JPEG/PNG/TIFF preservando precisión de 16 bits cuando procede."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError:
            return _read_with_pillow(path, ignore_icc)
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            photometric = str(page.photometric).upper()
            if "PALETTE" in photometric or "SEPARATED" in photometric or "YCBCR" in photometric:
                return _read_with_pillow(path, ignore_icc)
            array = page.asarray()
            icc = page.tags.get(34675)
            if icc is not None and not ignore_icc:
                if array.dtype == np.uint8:
                    return _read_with_pillow(path, ignore_icc)
                raise ConversionError(
                    "TIFF de alta profundidad con ICC: convertir previamente a TIFF RGB lineal o usar --ignore-icc"
                )
            rgb, alpha = _image_channels(array)
            if rgb.dtype.kind not in "uf":
                raise ConversionError(f"Tipo de TIFF incompatible: {rgb.dtype}")
            if rgb.dtype.kind == "f":
                if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                    raise ConversionError("TIFF float requiere valores normalizados en [0,1]")
                scale = None
            else:
                scale = float(np.iinfo(rgb.dtype).max)
            return LoadedImage(
                rgb,
                alpha,
                str(rgb.dtype),
                scale,
                "tifffile",
                "ignored_by_request" if icc is not None else "absent",
                ["TIFF >= 16 bits: especificar --input-encoding cuando exista duda sobre gamma"],
            )
    if suffix == ".png" and _read_png_bit_depth(path) == 16:
        try:
            import cv2
        except ImportError as exc:
            raise ConversionError("PNG RGB 16-bit requiere opencv-python para preservar 16 bits") from exc
        x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if x is None:
            raise ConversionError("No se ha podido abrir el PNG de 16 bits")
        if x.ndim == 3 and x.shape[2] == 3:
            x = x[:, :, ::-1].copy()
        elif x.ndim == 3 and x.shape[2] == 4:
            x = x[:, :, [2, 1, 0, 3]]
        rgb, alpha = _image_channels(x)
        with Image.open(path) as im:
            has_icc = bool(im.info.get("icc_profile"))
        if has_icc and not ignore_icc:
            raise ConversionError(
                "PNG 16-bit con ICC: convertir primero a sRGB lineal o usar --ignore-icc"
            )
        return LoadedImage(
            rgb, alpha, str(rgb.dtype), 65535.0, "opencv_png16",
            "ignored_by_request" if has_icc else "absent", [],
        )
    return _read_with_pillow(path, ignore_icc)


def _resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = array.shape[:2]
    if (src_w, src_h) == (width, height):
        return array
    if array.dtype == np.uint8:
        mode = "RGB" if array.ndim == 3 and array.shape[-1] == 3 else "L"
        if array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 3):
            pil = Image.fromarray(array, mode=mode)
            resampling = (
                Image.Resampling.BOX
                if width < src_w and height < src_h
                else Image.Resampling.BICUBIC
            )
            return np.asarray(pil.resize((width, height), resample=resampling))
    try:
        import cv2
    except ImportError as exc:
        raise ConversionError("El reescalado de TIFF/PNG 16-bit requiere opencv-python") from exc
    method = cv2.INTER_AREA if width < src_w or height < src_h else cv2.INTER_LINEAR
    return cv2.resize(array, (width, height), interpolation=method)


def _layout(src_w: int, src_h: int, dst_w: int, dst_h: int, mode: str) -> dict[str, int | str | bool]:
    """Calcula un encaje proporcional sin recortar entre orientaciones distintas."""
    requested_mode = mode
    orientation_mismatch = (
        src_w != src_h
        and dst_w != dst_h
        and (src_w > src_h) != (dst_w > dst_h)
    )
    # Cubrir y recortar una imagen vertical para un sensor horizontal (o al
    # revés) descarta casi toda la imagen. Ese ajuste corresponde a
    # --orientation match-sensor; mientras no se solicite, conservamos la
    # imagen completa con el comportamiento contain.
    if mode == "cover" and orientation_mismatch:
        mode = "contain"

    if mode == "none":
        if (src_w, src_h) != (dst_w, dst_h):
            raise ConversionError("--resize none exige dimensiones exactamente iguales al RAW")
        return {
            "new_width": dst_w,
            "new_height": dst_h,
            "x": 0,
            "y": 0,
            "crop_x": 0,
            "crop_y": 0,
            "mode": mode,
            "requested_mode": requested_mode,
            "orientation_mismatch": orientation_mismatch,
        }
    if mode == "stretch":
        return {
            "new_width": dst_w,
            "new_height": dst_h,
            "x": 0,
            "y": 0,
            "crop_x": 0,
            "crop_y": 0,
            "mode": mode,
            "requested_mode": requested_mode,
            "orientation_mismatch": orientation_mismatch,
        }
    scale = (min if mode == "contain" else max)(dst_w / src_w, dst_h / src_h)
    w, h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    if mode == "contain":
        w, h = min(w, dst_w), min(h, dst_h)
        x, y = (dst_w - w) // 2, (dst_h - h) // 2
        return {
            "new_width": w,
            "new_height": h,
            "x": x,
            "y": y,
            "crop_x": 0,
            "crop_y": 0,
            "mode": mode,
            "requested_mode": requested_mode,
            "orientation_mismatch": orientation_mismatch,
        }
    w, h = max(w, dst_w), max(h, dst_h)
    return {
        "new_width": w,
        "new_height": h,
        "x": 0,
        "y": 0,
        "crop_x": (w - dst_w) // 2,
        "crop_y": (h - dst_h) // 2,
        "mode": mode,
        "requested_mode": requested_mode,
        "orientation_mismatch": orientation_mismatch,
    }


def prepare_preview_rgb8(
    source: str | Path,
    *,
    target_width: int,
    target_height: int,
    resize: str = "cover",
    orientation: str = "preserve",
    rotation: str | None = None,
    input_encoding: str = "auto",
    background_linear: float = 0.18,
    exposure_ev: float = 0.0,
    ignore_icc: bool = False,
    precomputed_source_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepara una imagen sRGB de 8 bits con la geometría visible del RAW.

    Esta ruta está destinada a las vistas previas TIFF/JPEG del contenedor. Comparte
    con la síntesis CFA la orientación, el encuadre, el aplanado del alfa y la
    compensación EV, pero no aplica la matriz RGB->cámara ni los artefactos del
    sensor: esos pasos no tienen una conversión JPEG de cámara inequívoca.
    """
    if target_width <= 0 or target_height <= 0:
        raise ConversionError("Las dimensiones objetivo del preview deben ser positivas")
    if resize not in ("contain", "cover", "stretch", "none"):
        raise ConversionError("resize debe ser contain, cover, stretch o none")
    if input_encoding not in ("auto", "srgb", "linear"):
        raise ConversionError("input_encoding debe ser auto, srgb o linear")
    if not 0.0 <= background_linear <= 1.0:
        raise ConversionError("background_linear fuera de [0,1]")
    if not math.isfinite(exposure_ev) or not -12 <= exposure_ev <= 12:
        raise ConversionError("exposure_ev fuera de [-12,12]")

    src = Path(source)
    loaded = load_image(src, ignore_icc=ignore_icc)
    loaded, orientation_info = _orient_loaded_image(
        loaded,
        target_width=target_width,
        target_height=target_height,
        orientation=orientation,
        rotation=rotation,
    )
    h0, w0 = loaded.rgb.shape[:2]
    mapping = _layout(w0, h0, target_width, target_height, resize)
    w, h = int(mapping["new_width"]), int(mapping["new_height"])
    rgb = _resize(loaded.rgb, w, h)
    alpha = _resize(loaded.alpha, w, h) if loaded.alpha is not None else None
    crop_x, crop_y = int(mapping["crop_x"]), int(mapping["crop_y"])
    if crop_x or crop_y:
        rgb = rgb[crop_y : crop_y + target_height, crop_x : crop_x + target_width]
        if alpha is not None:
            alpha = alpha[crop_y : crop_y + target_height, crop_x : crop_x + target_width]
        w, h = rgb.shape[1], rgb.shape[0]

    max_in = loaded.source_max
    if max_in is None:
        if rgb.dtype.kind != "f":
            raise ConversionError("Valores enteros sin rango declarado")
        max_in = 1.0
    encoding = input_encoding
    if encoding == "auto":
        encoding = "srgb" if loaded.rgb.dtype == np.uint8 else "linear"

    linear = rgb.astype(np.float32) / np.float32(max_in)
    linear = np.clip(linear, 0.0, 1.0)
    if encoding == "srgb":
        linear = _eotf_srgb(linear).astype(np.float32, copy=False)
    if alpha is not None:
        a = alpha.astype(np.float32)
        alpha_max = float(np.iinfo(alpha.dtype).max) if alpha.dtype.kind in "ui" else 1.0
        a = np.clip(a / alpha_max, 0.0, 1.0)[..., None]
        linear = linear * a + np.float32(background_linear) * (1.0 - a)

    canvas = np.full(
        (target_height, target_width, 3),
        np.float32(background_linear),
        dtype=np.float32,
    )
    x0, y0 = int(mapping["x"]), int(mapping["y"])
    canvas[y0 : y0 + h, x0 : x0 + w] = linear
    canvas *= np.float32(2.0**exposure_ev)
    output = np.rint(_oetf_srgb(canvas) * np.float32(255.0)).astype(np.uint8)
    source_sha256 = (
        precomputed_source_sha256
        if precomputed_source_sha256 is not None
        else sha256_file(src)
    )
    return output, {
        "source": {
            "filename": src.name,
            "sha256": source_sha256,
            "shape_hw": orientation_info["source_shape_hw"],
            "dtype": loaded.source_dtype,
            "backend": loaded.backend,
            "icc_handling": loaded.icc_handling,
        },
        "target": {"width": target_width, "height": target_height},
        "processing": {
            "resize": mapping,
            "orientation": orientation_info,
            "input_encoding_used": encoding,
            "background_linear": background_linear,
            "exposure_ev": exposure_ev,
            "alpha_flattened": alpha is not None,
            "notes": loaded.notes,
        },
        "output": {"dtype": "uint8", "shape_hw": [target_height, target_width]},
    }


class ImageToBayerConverter:
    """Sintetizador vectorizado sRGB/lineal -> mosaico CFA Bayer uint16 de 14 bits."""

    def __init__(self, target: BayerTarget, config: BayerConversionConfig | None = None) -> None:
        self.target = target
        self.config = config or BayerConversionConfig()
        self.config.validate(target)

    def synthesize_in_memory(
        self,
        source: str | Path,
        *,
        precomputed_source_sha256: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sintetiza el mosaico Bayer 2D uint16 íntegramente en memoria (~0.45 s) y devuelve
        (array_uint16, manifest_dict_parcial).
        """
        src = Path(source)
        loaded = load_image(src, ignore_icc=self.config.ignore_icc)
        loaded, orientation_info = _orient_loaded_image(
            loaded,
            target_width=self.target.width,
            target_height=self.target.height,
            orientation=self.config.orientation,
            rotation=self.config.rotation,
        )
        h0, w0 = loaded.rgb.shape[:2]
        mapping = _layout(w0, h0, self.target.width, self.target.height, self.config.resize)
        w, h = int(mapping["new_width"]), int(mapping["new_height"])
        rgb = _resize(loaded.rgb, w, h)
        alpha = _resize(loaded.alpha, w, h) if loaded.alpha is not None else None
        crop_x, crop_y = int(mapping["crop_x"]), int(mapping["crop_y"])
        if crop_x or crop_y:
            rgb = rgb[crop_y : crop_y + self.target.height, crop_x : crop_x + self.target.width]
            if alpha is not None:
                alpha = alpha[crop_y : crop_y + self.target.height, crop_x : crop_x + self.target.width]
            w, h = rgb.shape[1], rgb.shape[0]
        x0, y0 = int(mapping["x"]), int(mapping["y"])
        max_in = loaded.source_max
        if max_in is None:
            if rgb.dtype.kind != "f":
                raise ConversionError("Valores enteros sin rango declarado")
            max_in = 1.0
        encoding = self.config.input_encoding
        if encoding == "auto":
            encoding = "srgb" if loaded.rgb.dtype == np.uint8 else "linear"

        gain_map = {
            "R": float(self.config.gains[0]),
            "G1": float(self.config.gains[1]),
            "G2": float(self.config.gains[2]),
            "B": float(self.config.gains[3]),
        }
        matrix = np.asarray(self.config.rgb_to_camera, dtype=np.float32)
        cfa = self.target.cfa
        exposure = np.float32(2.0**self.config.exposure_ev)
        black = self.config.black_dn
        span = self.config.white_dn - self.config.black_dn
        src_sha = precomputed_source_sha256 if precomputed_source_sha256 is not None else sha256_file(src)

        out = np.empty((self.target.height, self.target.width), dtype=np.uint16)
        bg = np.float32(self.config.background_linear) * exposure
        greens = [(i // 2, i % 2) for i, letter in enumerate(cfa) if letter == "G"]

        # Rellenar el fondo neutro solo en los márgenes si la imagen no cubre todo el sensor
        has_margins = x0 > 0 or y0 > 0 or (x0 + w) < self.target.width or (y0 + h) < self.target.height
        if has_margins:
            for ry in (0, 1):
                for cx in (0, 1):
                    ch = cfa[ry * 2 + cx]
                    if ch == "G":
                        channel = "G1" if (ry, cx) == greens[0] else "G2"
                        mi = 1
                    else:
                        mi = 0 if ch == "R" else 2
                        channel = ch
                    g = gain_map[channel]
                    effective = bg * float(matrix[mi].sum()) * g
                    dn = int(np.clip(np.rint(black + span * effective), black, self.config.white_dn))
                    if y0 > 0:
                        out[ry:y0:2, cx::2] = dn
                    if y0 + h < self.target.height:
                        start_y = y0 + h + ((ry - (y0 + h)) % 2)
                        out[start_y::2, cx::2] = dn
                    if x0 > 0:
                        start_y = y0 + ((ry - y0) % 2)
                        out[start_y : y0 + h : 2, cx:x0:2] = dn
                    if x0 + w < self.target.width:
                        start_y = y0 + ((ry - y0) % 2)
                        start_x = x0 + w + ((cx - (x0 + w)) % 2)
                        out[start_y : y0 + h : 2, start_x::2] = dn

        # Ruta rápida vectorizada por LUT de 256 entradas para imágenes uint8 sin canal alfa
        use_u8_lut = rgb.dtype == np.uint8 and alpha is None and max_in == 255.0
        if use_u8_lut:
            lut_exp = (
                (_SRGB_U8_LUT_F32 * exposure).astype(np.float32)
                if encoding == "srgb"
                else (_SRGB_U8_NORM_F32 * exposure).astype(np.float32)
            )
            for ry in (0, 1):
                sy = (ry - y0) % 2
                if sy >= h:
                    continue
                for cx in (0, 1):
                    sx = (cx - x0) % 2
                    if sx >= w:
                        continue
                    letter = cfa[ry * 2 + cx]
                    if letter == "G":
                        gn = "G1" if (ry, cx) == greens[0] else "G2"
                        mi = 1
                    else:
                        gn = letter
                        mi = 0 if letter == "R" else 2
                    sub_u8 = rgb[sy::2, sx::2]
                    b0 = lut_exp[sub_u8[..., 0]]
                    b1 = lut_exp[sub_u8[..., 1]]
                    b2 = lut_exp[sub_u8[..., 2]]
                    sensor = b0 * matrix[mi, 0] + b1 * matrix[mi, 1] + b2 * matrix[mi, 2]
                    dn = np.clip(
                        np.rint(black + span * sensor * gain_map[gn]),
                        black,
                        self.config.white_dn,
                    )
                    out[y0 + sy : y0 + h : 2, x0 + sx : x0 + w : 2] = dn.astype(np.uint16)
        else:
            for iy in range(0, h, 256):
                block = rgb[iy : iy + 256].astype(np.float32) / np.float32(max_in)
                block = np.clip(block, 0.0, 1.0)
                if encoding == "srgb":
                    block = _eotf_srgb(block)
                if alpha is not None:
                    a = alpha[iy : iy + 256].astype(np.float32)
                    alpha_max = float(np.iinfo(alpha.dtype).max) if alpha.dtype.kind in "ui" else 1.0
                    a = np.clip(a / alpha_max, 0.0, 1.0)[..., None]
                    block = block * a + np.float32(self.config.background_linear) * (1.0 - a)
                block *= exposure
                for ry in (0, 1):
                    global_y = y0 + iy
                    sy = (ry - global_y) % 2
                    if sy >= block.shape[0]:
                        continue
                    for cx in (0, 1):
                        sx = (cx - x0) % 2
                        if sx >= block.shape[1]:
                            continue
                        letter = cfa[ry * 2 + cx]
                        if letter == "G":
                            gn = "G1" if (ry, cx) == greens[0] else "G2"
                            mi = 1
                        else:
                            gn = letter
                            mi = 0 if letter == "R" else 2
                        b = block[sy::2, sx::2]
                        sensor = (
                            b[..., 0] * matrix[mi, 0]
                            + b[..., 1] * matrix[mi, 1]
                            + b[..., 2] * matrix[mi, 2]
                        )
                        dn = np.clip(
                            np.rint(black + span * sensor * gain_map[gn]),
                            black,
                            self.config.white_dn,
                        )
                        out[
                            y0 + iy + sy : y0 + iy + block.shape[0] : 2,
                            x0 + sx : x0 + block.shape[1] : 2,
                        ] = dn.astype(np.uint16)

        minimum, maximum = int(out.min()), int(out.max())
        info: dict[str, Any] = {
            "schema_version": "1.0.0",
            "tool": {"name": "nefinject.synthesis", "version": VERSION},
            "type": "synthetic_cfa_raw",
            "source": {
                "filename": src.name,
                "sha256": src_sha,
                "shape_hw": orientation_info["source_shape_hw"],
                "dtype": loaded.source_dtype,
                "backend": loaded.backend,
                "icc_handling": loaded.icc_handling,
            },
            "target": {
                "width": self.target.width,
                "height": self.target.height,
                "cfa": self.target.cfa,
                "bits_per_sample": self.target.bit_depth,
                "reference_nef_sha256": self.target.reference_sha256,
                "reference_model": self.target.source_model,
            },
            "processing": {
                "resize": mapping,
                "orientation": orientation_info,
                "input_encoding_used": encoding,
                "black_dn": black,
                "white_dn": self.config.white_dn,
                "background_linear": self.config.background_linear,
                "exposure_ev": self.config.exposure_ev,
                "gains_r_g1_g2_b": list(self.config.gains),
                "rgb_to_camera_matrix": [list(row) for row in self.config.rgb_to_camera],
                "camera_model_calibrated": False,
                "optical_psf_applied": False,
                "notes": loaded.notes,
            },
            "array": {
                "filename": "bayer_final.npy",
                "shape_hw": [self.target.height, self.target.width],
                "dtype": "uint16",
                "min_dn": minimum,
                "max_dn": maximum,
            },
            "limits": [
                "RGB->sensor por matriz es una aproximación tricromática, no una simulación espectral Nikon calibrada.",
                "El RAW no incorpora efectos OLPF, difusión, ruido ni integración física del fotosito.",
                "Al reescalar una carta se modifica su frecuencia espacial; mantener escala 1:1 para ensayos de muaré.",
                "El perfil de color, exposición y balance de blancos aplicados por reveladores pueden alterar la apariencia.",
            ],
        }
        return out, info

    def synthesize_from_linear_rgb(
        self,
        rgb_linear: np.ndarray,
        *,
        source_filename: str = "optica_pre_cfa_linear.tif",
        source_sha256: str = "",
        optical_psf_applied: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sintetiza el mosaico Bayer 2D uint16 directamente desde un array RGB lineal float32 [0, 1]
        en memoria (ej. salida de apply_optics / simulate_optics_in_memory) sin pasar por un TIFF de 294 MB.
        """
        arr = np.asarray(rgb_linear, dtype=np.float32)
        if arr.ndim != 3 or arr.shape != (self.target.height, self.target.width, 3):
            raise ConversionError(
                f"Se esperaba RGB lineal float32 {(self.target.height, self.target.width, 3)}, "
                f"recibido {arr.shape}"
            )
        gain_map = {
            "R": float(self.config.gains[0]),
            "G1": float(self.config.gains[1]),
            "G2": float(self.config.gains[2]),
            "B": float(self.config.gains[3]),
        }
        matrix = np.asarray(self.config.rgb_to_camera, dtype=np.float32)
        cfa = self.target.cfa
        exposure = np.float32(2.0**self.config.exposure_ev)
        black = self.config.black_dn
        span = self.config.white_dn - self.config.black_dn
        greens = [(i // 2, i % 2) for i, letter in enumerate(cfa) if letter == "G"]

        out = np.empty((self.target.height, self.target.width), dtype=np.uint16)
        for ry in (0, 1):
            for cx in (0, 1):
                letter = cfa[ry * 2 + cx]
                if letter == "G":
                    gn = "G1" if (ry, cx) == greens[0] else "G2"
                    mi = 1
                else:
                    gn = letter
                    mi = 0 if letter == "R" else 2
                sub = arr[ry::2, cx::2] * exposure
                sensor = (
                    sub[..., 0] * matrix[mi, 0]
                    + sub[..., 1] * matrix[mi, 1]
                    + sub[..., 2] * matrix[mi, 2]
                )
                dn = np.clip(
                    np.rint(black + span * sensor * gain_map[gn]),
                    black,
                    self.config.white_dn,
                )
                out[ry::2, cx::2] = dn.astype(np.uint16)

        minimum, maximum = int(out.min()), int(out.max())
        mapping = _layout(
            self.target.width, self.target.height, self.target.width, self.target.height, "none"
        )
        info: dict[str, Any] = {
            "schema_version": "1.0.0",
            "tool": {"name": "nefinject.synthesis", "version": VERSION},
            "type": "synthetic_cfa_raw",
            "source": {
                "filename": source_filename,
                "sha256": source_sha256,
                "shape_hw": [self.target.height, self.target.width],
                "dtype": "float32",
                "backend": "in_memory_optics",
                "icc_handling": "absent",
            },
            "target": {
                "width": self.target.width,
                "height": self.target.height,
                "cfa": self.target.cfa,
                "bits_per_sample": self.target.bit_depth,
                "reference_nef_sha256": self.target.reference_sha256,
                "reference_model": self.target.source_model,
            },
            "processing": {
                "resize": mapping,
                "input_encoding_used": "linear",
                "black_dn": black,
                "white_dn": self.config.white_dn,
                "background_linear": self.config.background_linear,
                "exposure_ev": self.config.exposure_ev,
                "gains_r_g1_g2_b": list(self.config.gains),
                "rgb_to_camera_matrix": [list(row) for row in self.config.rgb_to_camera],
                "camera_model_calibrated": False,
                "optical_psf_applied": optical_psf_applied,
                "notes": ["Sintetizado en RAM desde RGB lineal float32 post-óptica"],
            },
            "array": {
                "filename": "bayer_final.npy",
                "shape_hw": [self.target.height, self.target.width],
                "dtype": "uint16",
                "min_dn": minimum,
                "max_dn": maximum,
            },
            "limits": [
                "RGB->sensor por matriz es una aproximación tricromática, no una simulación espectral Nikon calibrada.",
                "El perfil de color, exposición y balance de blancos aplicados por reveladores pueden alterar la apariencia.",
            ],
        }
        return out, info

    def convert(
        self,
        source: str | Path,
        output_npy: str | Path,
        *,
        create_npz: bool = False,
        overwrite: bool = False,
        precomputed_source_sha256: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sintetiza el mosaico Bayer, lo persiste en output_npy + manifiesto .json y devuelve (array, info)."""
        src, dest = Path(source), Path(output_npy)
        if dest.suffix.lower() != ".npy":
            raise ConversionError("La salida principal debe ser .npy")
        manifest = dest.with_suffix(".json")
        npz = dest.with_suffix(".npz")
        reserved = [dest, manifest] + ([npz] if create_npz else [])
        if len({p.resolve() for p in reserved + [src]}) != len(reserved) + 1:
            raise ConversionError("Las rutas de entrada y salida no pueden coincidir")
        if not overwrite and any(p.exists() for p in reserved):
            raise FileExistsError("Existe algún archivo de salida: no se sobrescribe")

        out, info = self.synthesize_in_memory(src, precomputed_source_sha256=precomputed_source_sha256)
        info["array"]["filename"] = dest.name
        npy_sha = save_npy_and_sha256(dest, out, overwrite=overwrite)
        info["array"]["sha256"] = npy_sha
        if create_npz:
            np.savez_compressed(npz, raw=out)
            info["archive"] = {"filename": npz.name, "key": "raw", "sha256": sha256_file(npz)}
        write_json_atomic(manifest, info, overwrite=overwrite)
        return out, info


# ---------------------------------------------------------------------------
# Generador de Patrones de Frecuencia Espacial (Chirp / Flat)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BayerPatternConfig:
    width: int = 6080
    height: int = 4028
    bit_depth: int = 14
    cfa: str = "RGGB"
    pattern_type: str = "chirp"
    mean_dn: float = 7000.0
    amplitude_dn: float = 3500.0
    start_cycles_per_pixel: float = 0.02
    end_cycles_per_pixel: float = 0.75
    phase_cycles: float = 0.0

    def validate(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Las dimensiones deben ser >= 2")
        if self.bit_depth != 14:
            raise ValueError("El codificador Nikon D610 solo admite 14 bits")
        if self.cfa != "RGGB":
            raise ValueError("Para este ensayo solo se admite CFA RGGB")
        if self.pattern_type not in ("chirp", "flat"):
            raise ValueError("Tipo no soportado: elija chirp o flat")
        white = (1 << self.bit_depth) - 1
        if self.mean_dn - self.amplitude_dn < 0 or self.mean_dn + self.amplitude_dn > white:
            raise ValueError(f"Señal fuera del intervalo de 14 bits [0,{white}]")
        if not (0 <= self.start_cycles_per_pixel <= self.end_cycles_per_pixel <= 1):
            raise ValueError("El chirp debe utilizar 0 <= f0 <= f1 <= 1 ciclos/fotosito")


class BayerPatternGenerator:
    """Genera un mosaico 2D uint16 analítico (chirp de fase integrada o plano)."""

    def __init__(self, config: BayerPatternConfig) -> None:
        config.validate()
        self.config = config

    def row(self) -> np.ndarray:
        c = self.config
        if c.pattern_type == "flat":
            return np.full(c.width, round(c.mean_dn), dtype=np.uint16)
        x = np.arange(c.width, dtype=np.float64)
        k = (c.end_cycles_per_pixel - c.start_cycles_per_pixel) / (c.width - 1)
        phase = c.phase_cycles + c.start_cycles_per_pixel * x + 0.5 * k * x * x
        signal = c.mean_dn + c.amplitude_dn * np.sin(2.0 * np.pi * phase)
        return np.clip(np.rint(signal), 0, (1 << c.bit_depth) - 1).astype(np.uint16)

    def generate_in_memory(self) -> np.ndarray:
        c = self.config
        line = self.row()
        return np.broadcast_to(line, (c.height, c.width)).copy()

    def save(
        self,
        destination: str | Path,
        *,
        force: bool = False,
        reference_sha256: str | None = None,
    ) -> dict[str, Any]:
        path = Path(destination)
        c = self.config
        arr = self.generate_in_memory()
        digest = save_npy_and_sha256(path, arr, overwrite=force)
        info = {
            "schema_version": "1.0.0",
            "type": "synthetic_cfa_raw",
            "description": "Un valor por fotosito, antes del demosaicing; chirp acromatico",
            "parameters": asdict(c),
            "array": {
                "filename": path.name,
                "shape_hw": [c.height, c.width],
                "dtype": "uint16",
                "minimum_dn": int(arr[0].min()),
                "maximum_dn": int(arr[0].max()),
                "bytes": path.stat().st_size,
                "sha256": digest,
            },
            "reference_nef_sha256": reference_sha256,
            "notes": [
                "La imagen RAW contiene exactamente un valor por fotosito: no es RGB ni imagen demosaicada.",
                "La señal es acromatica: los valores de R, G y B se muestrean del mismo campo escalar.",
                "Las frecuencias > 0.5 cy/px superan Nyquist de la rejilla y pueden plegarse por aliasing.",
                "No incluye PSF/OLPF, respuesta espectral, ruido, ganancia de canal ni caracterizacion D610.",
                "El fichero NPY es una entrada del codificador, no un NEF comercial listo para abrir.",
            ],
        }
        write_json_atomic(path.with_suffix(".json"), info, overwrite=force)
        return info


# ---------------------------------------------------------------------------
# Carta Macbeth ColorChecker 24 Parches y Layout de ROIs
# ---------------------------------------------------------------------------

COLORCHECKER_24_SRGB: tuple[tuple[int, int, int], ...] = (
    (115, 82, 68),   (194, 150, 130), (98, 122, 157),  (87, 108, 67),   (133, 128, 177), (103, 189, 170),
    (214, 126, 44),  (80, 91, 166),   (193, 90, 99),   (94, 60, 108),   (157, 188, 64),  (224, 163, 46),
    (56, 61, 150),   (70, 148, 73),   (175, 54, 60),   (231, 199, 31),  (187, 86, 149),  (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85),    (52, 52, 52),
)


def generate_colorchecker24_png(
    output_path: str | Path,
    *,
    width: int = 1800,
    height: int = 1200,
    cols: int = 6,
    rows: int = 4,
) -> Path:
    """Genera una imagen PNG sintética sRGB de 8 bits con los 24 parches Macbeth ColorChecker."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cell_w = width / cols
    cell_h = height / rows
    for idx, rgb in enumerate(COLORCHECKER_24_SRGB):
        r, c = divmod(idx, cols)
        y0, y1 = int(round(r * cell_h)), int(round((r + 1) * cell_h))
        x0, x1 = int(round(c * cell_w)), int(round((c + 1) * cell_w))
        canvas[y0:y1, x0:x1] = rgb
    Image.fromarray(canvas, mode="RGB").save(dest, format="PNG")
    return dest


def generate_siemens_star_png(
    output_path: str | Path,
    *,
    width: int = 1800,
    height: int = 1200,
    cycles: int = 144,
    center_radius: float = 24.0,
    outer_radius: float | None = None,
) -> Path:
    """Genera una estrella Siemens sinusoidal sRGB para evaluar resolución y muaré.

    ``cycles`` expresa el número de ciclos completos alrededor del centro. Las
    zonas interior y exterior se dejan en gris medio para delimitar el patrón.
    No pretende ser una carta ISO 12233 completa.
    """
    if width < 3 or height < 3:
        raise ValueError("La estrella Siemens requiere al menos 3x3 píxeles")
    if cycles < 1 or center_radius < 0 or not np.isfinite(center_radius):
        raise ValueError("cycles y center_radius no son válidos")

    default_outer_radius = min(width, height) / 2.0 - 10.0
    radius = default_outer_radius if outer_radius is None else outer_radius
    if not np.isfinite(radius) or radius <= center_radius:
        raise ValueError("outer_radius debe ser mayor que center_radius")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((height, width), dtype=np.float32)
    dx = x - (width - 1) / 2.0
    dy = y - (height - 1) / 2.0
    distance = np.hypot(dx, dy)
    valid = (distance >= center_radius) & (distance <= radius)

    image = np.full((height, width), 128, dtype=np.uint8)
    theta = np.arctan2(dy[valid], dx[valid])
    image[valid] = np.rint(127.5 + 127.5 * np.sin(cycles * theta)).astype(np.uint8)
    rgb = np.repeat(image[..., np.newaxis], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(dest, format="PNG")
    return dest


def make_grid_layout(
    cols: int = 6,
    rows: int = 4,
    fill: float = 0.5,
    neutral_last_row: bool = True,
) -> dict[str, Any]:
    """Construye la plantilla de ROIs relativas para una carta regular cols x rows."""
    if cols < 1 or rows < 1 or not 0.05 <= fill <= 0.95:
        raise ValueError("cols/rows positivos, fill en [0.05, 0.95]")
    patches = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            dx = fill / (2.0 * cols)
            dy = fill / (2.0 * rows)
            patches.append(
                {
                    "id": f'{chr(65 + row) if row < 26 else "R" + str(row + 1)}{col + 1:02d}',
                    "role": "neutral" if neutral_last_row and row == rows - 1 else "chromatic",
                    "reference_roi": [x - dx, y - dy, x + dx, y + dy],
                }
            )
    return {
        "schema_version": "1.0.0",
        "roi_units": "relative",
        "patches": patches,
        "note": "Ajustar rendered_roi y raw_roi por parche cuando el revelador recorte, rote o altere las proporciones de la carta.",
    }

