"""Simulador de degradación óptica pre-CFA basado en física de propagación de luz.

Forma parte de la librería `nefinject`, modelando las imperfecciones de la lente
y elementos ópticos del sistema de cámara en el dominio RGB lineal continuo (float32 [0, 1]),
antes de la discretización espacial y muestreo espectral del mosaico Bayer.

Modelos físicos integrados:
1. Función de Dispersión de Punto / Desenfoque (`PSF`):
   - Modelado gaussiano anisótropo (sigma_x, sigma_y) de difracción de apertura y aberración esférica.
2. Filtro Óptico Paso Bajo / Antialiasing (`OLPF`):
   - Desdoblamiento de haz birrefringente de 4 puntos (separación en X e Y) para atenuar
     frecuencias espaciales por encima del límite de Nyquist del sensor.
3. Aberración Cromática Lateral (`ChromaticAberration`):
   - Desplazamiento y magnificación radial diferencial de longitudes de onda (canales R y B
     escalados respecto al canal G desde el centro óptico).
4. Viñeteado Óptico (`Vignette`):
   - Atenuación de iluminancia radial según la ley del coseno a la cuarta cos^4(theta) y
     parámetros polinomiales de caída radial desde el punto principal.
5. Distorsión Geométrica Radial (`Distortion`):
   - Deformación de coordenadas por lente (barrilete / cojín) según modelo polinomial de Brown-Conrady.

Características de ejecución:
- Arquitectura 100% en memoria (`simulate_optics_in_memory`) para evitar escrituras intermedias
  en disco de imágenes descomprimidas (~294 MB por fotograma).
- Aceleración automática mediante primitivas optimizadas de OpenCV (`cv2`) con alternativa
  completa vectorizada en NumPy puro.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    NefInjectError,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from .synthesis import _SRGB_U8_LUT_F32, _eotf_srgb, _layout, _orient_loaded_image, load_image

VERSION = "1.0.0"


class OpticsError(NefInjectError):
    """Error de validación o ejecución en el simulador óptico pre-CFA."""


@dataclass(frozen=True)
class PSF:
    sigma_x_px: float = 0.0
    sigma_y_px: float = 0.0

    def validate(self) -> None:
        for x in (self.sigma_x_px, self.sigma_y_px):
            if not math.isfinite(x) or x < 0 or x > 30:
                raise ValueError("PSF sigma fuera de [0,30] px")


@dataclass(frozen=True)
class OLPF:
    separation_x_px: float = 0.0
    separation_y_px: float = 0.0

    def validate(self) -> None:
        for x in (self.separation_x_px, self.separation_y_px):
            if not math.isfinite(x) or x < 0 or x > 12:
                raise ValueError("OLPF separation fuera de [0,12] px")


@dataclass(frozen=True)
class Vignetting:
    edge_factor: float = 1.0
    power: float = 2.0
    center_x: float = 0.5
    center_y: float = 0.5

    def validate(self) -> None:
        if not (0 < self.edge_factor <= 1 and math.isfinite(self.edge_factor)):
            raise ValueError("vignetting.edge_factor fuera de (0,1]")
        if not (0.1 <= self.power <= 12 and math.isfinite(self.power)):
            raise ValueError("vignetting.power fuera de [0.1,12]")
        for x in (self.center_x, self.center_y):
            if not math.isfinite(x) or not (0 <= x <= 1):
                raise ValueError("Centro de viñeteado fuera de [0,1]")


@dataclass(frozen=True)
class LateralCA:
    """Desviación del radio de muestreo de R/B respecto a G (0 = identidad)."""

    red_scale_delta: float = 0.0
    blue_scale_delta: float = 0.0

    def validate(self) -> None:
        for x in (self.red_scale_delta, self.blue_scale_delta):
            if not math.isfinite(x) or abs(x) > 0.1:
                raise ValueError("CA lateral: escala fuera de +/-0.1")


@dataclass(frozen=True)
class Distortion:
    """Modelo polinómico sobre radio normalizado al semidiagonal: 1 + k1*r^2 + k2*r^4."""

    k1: float = 0.0
    k2: float = 0.0

    def validate(self) -> None:
        for x in (self.k1, self.k2):
            if not math.isfinite(x) or abs(x) > 0.7:
                raise ValueError("Distortion k fuera de +/-0.7")


@dataclass(frozen=True)
class OpticsConfig:
    psf: PSF = field(default_factory=PSF)
    olpf: OLPF = field(default_factory=OLPF)
    vignetting: Vignetting = field(default_factory=Vignetting)
    lateral_ca: LateralCA = field(default_factory=LateralCA)
    distortion: Distortion = field(default_factory=Distortion)
    background_linear: float = 0.18

    @classmethod
    def from_json(cls, file: str | Path) -> "OpticsConfig":
        d = json.loads(Path(file).read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("Optics JSON debe ser objeto")
        allowed = {
            "schema_version",
            "comment",
            "psf",
            "olpf",
            "vignetting",
            "lateral_ca",
            "distortion",
            "background_linear",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ValueError("Claves desconocidas Optics: " + str(sorted(unknown)))
        cfg = cls(
            psf=PSF(**d.get("psf", {})),
            olpf=OLPF(**d.get("olpf", {})),
            vignetting=Vignetting(**d.get("vignetting", {})),
            lateral_ca=LateralCA(**d.get("lateral_ca", {})),
            distortion=Distortion(**d.get("distortion", {})),
            background_linear=float(d.get("background_linear", 0.18)),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, spec: "OpticsConfig | str | Path") -> tuple["OpticsConfig", str]:
        """Carga un ``OpticsConfig`` desde una instancia o una ruta explícita a JSON."""
        if isinstance(spec, cls):
            spec.validate()
            return spec, "in_memory_dataclass"
        p = Path(str(spec))
        if p.suffix.lower() == ".json" and p.is_file():
            return cls.from_json(p), str(p.resolve())
        raise FileNotFoundError(
            f"No se encontró el perfil óptico JSON '{spec}'. "
            "Indica una ruta explícita a un archivo .json existente."
        )

    def validate(self) -> None:
        for obj in (self.psf, self.olpf, self.vignetting, self.lateral_ca, self.distortion):
            obj.validate()
        if not math.isfinite(self.background_linear) or not 0 <= self.background_linear <= 1:
            raise ValueError("background_linear debe estar entre 0 y 1")


def _maps(
    height: int,
    width: int,
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = np.float32(center_x * (width - 1))
    cy = np.float32(center_y * (height - 1))
    dx = xx - cx
    dy = yy - cy
    div = max((width - 1) / 2.0, 1.0) ** 2 + max((height - 1) / 2.0, 1.0) ** 2
    r2 = (dx * dx + dy * dy) / np.float32(div)
    return xx, yy, dx, dy, r2


def _reflect101_idx(idx: np.ndarray, size: int) -> np.ndarray:
    """Índice de reflexión BORDER_REFLECT_101 (dcb|abcdef|ed) en NumPy puro."""
    if size <= 1:
        return np.zeros_like(idx)
    period = 2 * (size - 1)
    m = np.mod(idx, period)
    return np.where(m < size, m, period - m)


def _remap_bilinear_numpy(img: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """Fallback de remapeo bilineal con BORDER_REFLECT_101 cuando OpenCV no está instalado."""
    h, w = img.shape[:2]
    x0 = np.floor(mx).astype(np.int32)
    y0 = np.floor(my).astype(np.int32)
    fx = (mx - x0).astype(np.float32)
    fy = (my - y0).astype(np.float32)
    x1 = _reflect101_idx(x0 + 1, w)
    y1 = _reflect101_idx(y0 + 1, h)
    x0 = _reflect101_idx(x0, w)
    y0 = _reflect101_idx(y0, h)
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]
    return (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy
    ).astype(np.float32, copy=False)


def _gaussian_1d_numpy(img: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma <= 0:
        return img
    ksize = int(max(3, 2 * math.ceil(3.0 * sigma) + 1))
    half = ksize // 2
    grid = np.arange(-half, half + 1, dtype=np.float32)
    ker = np.exp(-0.5 * (grid / np.float32(sigma)) ** 2)
    ker /= ker.sum()
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (half, half)
    padded = np.pad(img, pad_width, mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i, weight in enumerate(ker):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + img.shape[axis])
        out += padded[tuple(sl)] * weight
    return out


def _input_linear(
    path: Path,
    encoding: str,
    ignore_icc: bool,
    *,
    target_width: int,
    target_height: int,
    orientation: str,
    rotation: str | None,
) -> tuple[np.ndarray, Any, str, dict[str, Any]]:
    ld = load_image(path, ignore_icc=ignore_icc)
    ld, orientation_info = _orient_loaded_image(
        ld,
        target_width=target_width,
        target_height=target_height,
        orientation=orientation,
        rotation=rotation,
    )
    if ld.alpha is not None:
        alpha = np.clip(
            np.asarray(ld.alpha, dtype=np.float32) / np.float32(ld.source_max or 1.0),
            0.0,
            1.0,
        )
        rgb = np.asarray(ld.rgb, dtype=np.float32) * alpha[:, :, None]
        if ld.rgb.dtype.kind in "iu":
            rgb = np.rint(rgb).astype(ld.rgb.dtype)
        else:
            rgb = rgb.astype(ld.rgb.dtype, copy=False)
        ld.rgb = rgb
        ld.alpha = None
        ld.notes.append("Canal alpha aplanado sobre negro antes de la simulación óptica")
    if encoding not in ("srgb", "linear", "auto"):
        raise ValueError("Encoding invalido")
    enc = ("srgb" if ld.rgb.dtype == np.uint8 else "linear") if encoding == "auto" else encoding
    if ld.rgb.dtype == np.uint8 and (ld.source_max is None or ld.source_max == 255.0):
        if enc == "srgb":
            return _SRGB_U8_LUT_F32[ld.rgb], ld, enc, orientation_info
        return (ld.rgb.astype(np.float32) / np.float32(255.0)), ld, enc, orientation_info
    scale = float(ld.source_max) if ld.source_max is not None else 1.0
    x = np.asarray(ld.rgb, dtype=np.float32) / np.float32(scale)
    np.clip(x, 0.0, 1.0, out=x)
    if enc == "srgb":
        x = _eotf_srgb(x).astype(np.float32, copy=False)
    return x, ld, enc, orientation_info


def apply_optics(
    rgb_linear: np.ndarray,
    config: OpticsConfig,
    *,
    target_hw: tuple[int, int] | None = None,
    resize: str = "none",
) -> np.ndarray:
    """Aplica las degradaciones ópticas sobre una imagen RGB lineal float32 [0, 1].

    El desenfoque PSF y la separación OLPF se definen en píxeles de la rejilla final del sensor.
    Utiliza OpenCV (cv2) para un procesamiento acelerado de transformaciones y convoluciones, con
    fallback en NumPy puro si OpenCV no está instalado.
    """
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    config.validate()
    x = np.asarray(rgb_linear, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Entrada RGB lineal float32 HxWx3 normalizada [0,1]")
    src_h, src_w = x.shape[:2]
    if target_hw is not None:
        H, W = target_hw
        if H <= 1 or W <= 1:
            raise ValueError("Dimensiones invalidas")
        lay = _layout(src_w, src_h, W, H, resize)
        w, h = int(lay["new_width"]), int(lay["new_height"])
        if (w, h) != (src_w, src_h):
            if cv2 is not None:
                method = cv2.INTER_AREA if w < src_w or h < src_h else cv2.INTER_LINEAR
                x = cv2.resize(x, (w, h), interpolation=method)
            else:
                yy = np.linspace(0, src_h - 1, h, dtype=np.float32)
                xx = np.linspace(0, src_w - 1, w, dtype=np.float32)
                mx, my = np.meshgrid(xx, yy)
                x = _remap_bilinear_numpy(x, mx, my)
        crop_x, crop_y = int(lay["crop_x"]), int(lay["crop_y"])
        if crop_x or crop_y:
            x = x[crop_y : crop_y + H, crop_x : crop_x + W, :]
        if lay["mode"] == "contain":
            dest = np.empty((H, W, 3), dtype=np.float32)
            dest.fill(np.float32(config.background_linear))
            left, top = int(lay["x"]), int(lay["y"])
            dest[top : top + x.shape[0], left : left + x.shape[1], :] = x
            x = dest
        if x.shape != (H, W, 3):
            raise ValueError(f"Error geometria: {x.shape} != {(H, W, 3)}")

    H, W = x.shape[:2]
    lens = config.lateral_ca
    dist = config.distortion
    if any(abs(v) > 1e-12 for v in (lens.red_scale_delta, lens.blue_scale_delta, dist.k1, dist.k2)):
        xx, yy, dx, dy, r2 = _maps(H, W)
        radial = 1.0 + dist.k1 * r2 + dist.k2 * (r2 * r2)
        result = np.empty_like(x)
        for ch, ca in enumerate((lens.red_scale_delta, 0.0, lens.blue_scale_delta)):
            sc = radial * (1.0 + ca)
            mx = (xx + dx * (sc - 1.0)).astype(np.float32)
            my = (yy + dy * (sc - 1.0)).astype(np.float32)
            if cv2 is not None:
                result[:, :, ch] = cv2.remap(
                    x[:, :, ch], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
            else:
                result[:, :, ch] = _remap_bilinear_numpy(x[:, :, ch], mx, my)
        x = result
        del xx, yy, dx, dy, r2, radial, result

    if config.psf.sigma_x_px > 0 or config.psf.sigma_y_px > 0:
        if cv2 is not None:
            if config.psf.sigma_x_px > 0 and config.psf.sigma_y_px > 0:
                x = cv2.GaussianBlur(
                    x,
                    (0, 0),
                    sigmaX=config.psf.sigma_x_px,
                    sigmaY=config.psf.sigma_y_px,
                    borderType=cv2.BORDER_REFLECT_101,
                )
            else:
                for axis, sigma in ((0, config.psf.sigma_x_px), (1, config.psf.sigma_y_px)):
                    if sigma <= 0:
                        continue
                    ker = cv2.getGaussianKernel(int(max(3, 2 * math.ceil(3 * sigma) + 1)), sigma)
                    if axis == 0:
                        x = cv2.sepFilter2D(
                            x, -1, ker, np.array([1], dtype=np.float64), borderType=cv2.BORDER_REFLECT_101
                        )
                    else:
                        x = cv2.sepFilter2D(
                            x, -1, np.array([1], dtype=np.float64), ker, borderType=cv2.BORDER_REFLECT_101
                        )
        else:
            if config.psf.sigma_x_px > 0:
                x = _gaussian_1d_numpy(x, config.psf.sigma_x_px, axis=1)
            if config.psf.sigma_y_px > 0:
                x = _gaussian_1d_numpy(x, config.psf.sigma_y_px, axis=0)

    olpf = config.olpf
    if olpf.separation_x_px or olpf.separation_y_px:
        xx, yy, _, _, _ = _maps(H, W)
        accum = np.zeros_like(x)
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                mx = (xx + sx * olpf.separation_x_px).astype(np.float32)
                my = (yy + sy * olpf.separation_y_px).astype(np.float32)
                if cv2 is not None:
                    accum += (
                        cv2.remap(x, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) * 0.25
                    )
                else:
                    accum += _remap_bilinear_numpy(x, mx, my) * 0.25
        x = accum
        del xx, yy, accum

    vig = config.vignetting
    if vig.edge_factor < 1.0:
        _, _, _, _, r2 = _maps(H, W, center_x=vig.center_x, center_y=vig.center_y)
        r = np.minimum(1.0, np.sqrt(r2))
        transmission = 1.0 - (1.0 - vig.edge_factor) * np.power(r, vig.power)
        x = x * transmission[:, :, None]

    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def simulate_optics_in_memory(
    source: str | Path,
    config: OpticsConfig,
    *,
    width: int,
    height: int,
    resize: str = "contain",
    orientation: str = "preserve",
    rotation: str | None = None,
    encoding: str = "srgb",
    ignore_icc: bool = False,
    precomputed_source_sha256: str | None = None,
    output_filename_label: str = "in_memory_linear_rgb_float32",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ejecuta la simulación óptica pre-CFA íntegramente en RAM (sin escribir el TIFF de ~294 MB)
    y devuelve (rgb_linear_float32, report_dict).
    """
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(src)
    rgb, ld, enc, orientation_info = _input_linear(
        src,
        encoding,
        ignore_icc,
        target_width=width,
        target_height=height,
        orientation=orientation,
        rotation=rotation,
    )
    result = apply_optics(rgb, config, target_hw=(height, width), resize=resize)
    src_sha = precomputed_source_sha256 if precomputed_source_sha256 is not None else sha256_file(src)
    raw_bytes_sha = sha256_bytes(np.ascontiguousarray(result).view(np.uint8))
    report = {
        "schema_version": "1.0.0",
        "tool": {"name": "nefinject.optics", "version": VERSION},
        "source": {
            "filename": str(src),
            "sha256": src_sha,
            "shape_hw": orientation_info["source_shape_hw"],
            "dtype": ld.source_dtype,
            "icc_handling": ld.icc_handling,
        },
        "output": {
            "filename": output_filename_label,
            "sha256": raw_bytes_sha,
            "sha256_mode": "in_memory_float32_buffer",
            "shape_hw": [height, width],
            "dtype": "float32",
            "encoding": "linear RGB, sample values [0,1]",
        },
        "processing": {
            "resize": resize,
            "orientation": orientation_info,
            "source_encoding": enc,
            "optics": asdict(config),
            "order": [
                "RGB linearization",
                "resize to sensor grid",
                "distortion/CA lateral",
                "PSF convolution",
                "OLPF four shifted replicas",
                "vignetting",
            ],
        },
        "limitations": [
            "No representa PSF espectral o espacialmente variable medida de Nikon D610.",
            "La interpolacion del patron original afecta a las frecuencias para estudios de muare.",
            "El TIFF es RGB LINEAL, no sRGB con gamma. Pasar --input-encoding linear --resize none.",
            "La simulacion puede requerir RAM cercana a varios multiples del TIFF de salida.",
        ],
    }
    return result, report


def simulate_optics(
    source: str | Path,
    output: str | Path,
    config: OpticsConfig,
    *,
    width: int,
    height: int,
    resize: str = "contain",
    orientation: str = "preserve",
    rotation: str | None = None,
    encoding: str = "srgb",
    ignore_icc: bool = False,
) -> dict[str, Any]:
    """Ejecuta la simulación óptica y escribe el archivo TIFF RGB lineal float32 + manifiesto .json."""
    import tifffile

    src = Path(source)
    dst = Path(output)
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.resolve() == dst.resolve():
        raise ValueError("Entrada y salida no pueden ser iguales")
    if dst.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("La salida optica debe ser TIFF float32")
    if dst.exists() or dst.with_suffix(".json").exists():
        raise FileExistsError("Salida optica ya existente: usar otro nombre")
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgb, ld, enc, orientation_info = _input_linear(
        src,
        encoding,
        ignore_icc,
        target_width=width,
        target_height=height,
        orientation=orientation,
        rotation=rotation,
    )
    result = apply_optics(rgb, config, target_hw=(height, width), resize=resize)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".optics_", suffix=".tif", dir=dst.parent, delete=False
        ) as fd:
            tmp = Path(fd.name)
        tifffile.imwrite(tmp, result, photometric="rgb", metadata=None)
        with tifffile.TiffFile(tmp) as t:
            if tuple(t.pages[0].shape) != (height, width, 3) or t.pages[0].dtype != np.float32:
                raise ValueError("TIFF optico temporal invalido")
        report = {
            "schema_version": "1.0.0",
            "tool": {"name": "nefinject.optics", "version": VERSION},
            "source": {
                "filename": str(src),
                "sha256": sha256_file(src),
                "shape_hw": orientation_info["source_shape_hw"],
                "dtype": ld.source_dtype,
                "icc_handling": ld.icc_handling,
            },
            "output": {
                "filename": str(dst),
                "sha256": sha256_file(tmp),
                "shape_hw": [height, width],
                "dtype": "float32",
                "encoding": "linear RGB, sample values [0,1]",
            },
            "processing": {
                "resize": resize,
                "orientation": orientation_info,
                "source_encoding": enc,
                "optics": asdict(config),
                "order": [
                    "RGB linearization",
                    "resize to sensor grid",
                    "distortion/CA lateral",
                    "PSF convolution",
                    "OLPF four shifted replicas",
                    "vignetting",
                ],
            },
            "limitations": [
                "No representa PSF espectral o espacialmente variable medida de Nikon D610.",
                "La interpolacion del patron original afecta a las frecuencias para estudios de muare.",
                "El TIFF es RGB LINEAL, no sRGB con gamma. Pasar --input-encoding linear --resize none.",
                "La simulacion puede requerir RAM cercana a varios multiples del TIFF de salida.",
            ],
        }
        os.link(tmp, dst)
        try:
            write_json_atomic(dst.with_suffix(".json"), report, overwrite=False)
        except Exception:
            dst.unlink(missing_ok=True)
            raise
        return report
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

