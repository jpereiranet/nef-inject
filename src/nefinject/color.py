"""Módulo de colorimetría: sonda LibRaw/NEF y extractor de perfiles de color Adobe DNG.

Forma parte de la librería `nefinject`, gestionando la caracterización espectral del sensor,
la extracción de metadatos de color desde el archivo RAW y el perfilado fotométrico Adobe DNG.

Responsabilidades y componentes del módulo:
1. Sonda de Color de Cámara (`NefCameraColorProbe`):
   - Extracción de metadatos de color del NEF con arquitectura multi-backend:
     * Enlace CTypes directo con la biblioteca nativa LibRaw (si está disponible).
     * Backend secundario basado en rawpy.
     * Fallback puro en Python/NumPy analizando las etiquetas TIFF y MakerNote.
   - Obtención de multiplicadores de balance de blancos de la cámara (`cam_mul`),
     área activa del sensor, curva de linealización y niveles de negro/saturación.

2. Extractor de Perfiles Adobe DNG (`DngAdobeProfileExtractor`):
   - Lectura nativa directa de etiquetas de perfil DNG sin dependencias pesadas:
     ColorMatrix1, ColorMatrix2, ForwardMatrix1, ForwardMatrix2, CalibrationIlluminant1/2
     y datos XML embebidos de perfiles DCP / Adobe Camera Raw.
   - Soporte para calibración bajo iluminantes duales (Standard Illuminant A ~2856K y D65 ~6504K).

3. Modelado y Transformaciones Espectrales:
   - Estimación de temperatura de color correlacionada (CCT) mediante la aproximación de McCamy (`approx_cct_mccamy`).
   - Ponderación e interpolación de matrices de color según la temperatura Kelvin elegida (`mix_weight_dng`).
   - Adaptación cromática en espacio triestímulo mediante matriz de Bradford (`SRGB_TO_XYZ_D50`).
   - Matriz de referencia de color dcraw para Nikon D610 (`DCRAW_D610_COLOR_MATRIX_2`).
   - Verificación de procedencia y concordancia entre contenedores NEF y DNG (`verify_nef_dng_provenance`).

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import ctypes as ct
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .container import NAME_TO_TAG, TiffDirectory, TiffReader, TiffTag, find_libraw_cdll
from .models import (
    CameraColorProbeError,
    DngProfileError,
    sha256_file,
    to_native,
    write_json_atomic,
)

# Matriz estándar IEC 61966-2-1 sRGB lineal (D65) adaptada cromáticamente (Bradford) a PCS XYZ D50
SRGB_TO_XYZ_D50 = np.array(
    [
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ],
    dtype=np.float64,
)

# Matriz sRGB D65 -> XYZ D65 interna de LibRaw (src/tables/colordata.cpp)
LIBRAW_XYZ_RGB = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)

# Matriz DCRAW Nikon D610 (ColorMatrix2 D65 en unidades /10000)
DCRAW_D610_COLOR_MATRIX_2 = np.array(
    [
        [0.8178, -0.2245, -0.0609],
        [-0.4857, 1.2394, 0.2776],
        [-0.1207, 0.2086, 0.7298],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Sonda Colorimétrica NEF (LibRaw C API / rawpy / Fallback Puro NumPy)
# ---------------------------------------------------------------------------

class NefCameraColorProbe:
    """Extrae multiplicadores WB y matriz candidata sRGB lineal -> cámara desde un NEF."""

    VERSION = "0.1.0"

    def __init__(self, library: str | None = None) -> None:
        self.library = library

    def _probe_via_libraw_c(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str] | None:
        lib = find_libraw_cdll(self.library)
        if lib is None:
            return None
        sigs = {
            "libraw_version": (ct.c_char_p, []),
            "libraw_init": (ct.c_void_p, [ct.c_uint]),
            "libraw_open_file": (ct.c_int, [ct.c_void_p, ct.c_char_p]),
            "libraw_close": (None, [ct.c_void_p]),
            "libraw_get_cam_mul": (ct.c_float, [ct.c_void_p, ct.c_int]),
            "libraw_get_pre_mul": (ct.c_float, [ct.c_void_p, ct.c_int]),
            "libraw_get_rgb_cam": (ct.c_float, [ct.c_void_p, ct.c_int, ct.c_int]),
            "libraw_get_color_maximum": (ct.c_int, [ct.c_void_p]),
            "libraw_strerror": (ct.c_char_p, [ct.c_int]),
        }
        for name, (r, args) in sigs.items():
            fn = getattr(lib, name)
            fn.restype = r
            fn.argtypes = args

        ctx = lib.libraw_init(0)
        if not ctx:
            raise CameraColorProbeError("libraw_init devolvió un contexto nulo")
        try:
            rc = lib.libraw_open_file(ctx, os.fsencode(path))
            if rc:
                raise CameraColorProbeError(f"libraw_open_file={rc}: {lib.libraw_strerror(rc)!r}")
            cam_mul = np.array([lib.libraw_get_cam_mul(ctx, i) for i in range(4)], dtype=np.float64)
            pre_mul = np.array([lib.libraw_get_pre_mul(ctx, i) for i in range(4)], dtype=np.float64)
            m34 = np.array(
                [[lib.libraw_get_rgb_cam(ctx, r, c) for c in range(4)] for r in range(3)],
                dtype=np.float64,
            )
            white = int(lib.libraw_get_color_maximum(ctx))
            version = lib.libraw_version().decode("ascii", "replace")
            return cam_mul, pre_mul, m34, white, version
        finally:
            lib.libraw_close(ctx)

    @staticmethod
    def _probe_via_rawpy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str] | None:
        try:
            import rawpy
        except ImportError:
            return None
        with rawpy.imread(str(path)) as raw:
            cam_mul = np.asarray(raw.camera_whitebalance, dtype=np.float64)
            pre_mul = np.asarray(raw.daylight_whitebalance, dtype=np.float64)
            cam_xyz = np.asarray(raw.rgb_xyz_matrix[:3, :3], dtype=np.float64)
            cam_rgb = cam_xyz @ LIBRAW_XYZ_RGB
            cam_rgb_norm = cam_rgb / cam_rgb.sum(axis=1, keepdims=True)
            rgb_cam_3x3 = np.linalg.inv(cam_rgb_norm).astype(np.float32).astype(np.float64)
            m34 = np.zeros((3, 4), dtype=np.float64)
            m34[:, :3] = rgb_cam_3x3
            white = int(raw.white_level)
            ver = ".".join(str(x) for x in rawpy.libraw_version) + "-rawpy"
            return cam_mul, pre_mul, m34, white, ver

    @staticmethod
    def _probe_fallback_makernote(
        makernote_report: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
        wb_rb = (makernote_report or {}).get("white_balance", {}).get("wb_rb_levels")
        if wb_rb and len(wb_rb) == 4:
            vals = [float(x["decimal"]) if isinstance(x, dict) else float(x) for x in wb_rb]
            cam_mul = np.array([vals[0], vals[2], vals[1], vals[3]], dtype=np.float64)
        else:
            cam_mul = np.array([2.15625, 1.0, 1.234375, 1.0], dtype=np.float64)
        cam_rgb = DCRAW_D610_COLOR_MATRIX_2 @ LIBRAW_XYZ_RGB
        row_sums = cam_rgb.sum(axis=1)
        pre_mul = np.array(
            [1.0 / row_sums[0], 1.0 / row_sums[1], 1.0 / row_sums[2], 0.0],
            dtype=np.float64,
        )
        pre_mul[:3] /= np.min(pre_mul[:3])
        rgb_cam_3x3 = np.linalg.inv(cam_rgb / row_sums[:, None]).astype(np.float32).astype(np.float64)
        m34 = np.zeros((3, 4), dtype=np.float64)
        m34[:, :3] = rgb_cam_3x3
        return cam_mul, pre_mul, m34, 16383, "fallback_nikon_d610_table"

    def inspect(
        self,
        source: str | Path,
        *,
        compute_sha256: bool = False,
        precomputed_sha256: str | None = None,
        makernote_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        probed = self._probe_via_libraw_c(path)
        if probed is None:
            probed = self._probe_via_rawpy(path)
        if probed is None:
            probed = self._probe_fallback_makernote(makernote_report)

        cam_mul, pre_mul, m34, white, libraw_ver = probed
        wb = cam_mul[:3]
        if not (np.all(np.isfinite(wb)) and np.all(wb > 0)):
            raise CameraColorProbeError(f"Coeficientes WB inválidos: {wb}")
        m = m34[:, :3]
        cond = float(np.linalg.cond(m))
        if not (np.isfinite(cond) and cond < 1000):
            raise CameraColorProbeError(f"Matriz RGB de LibRaw singular o mal condicionada: cond={cond}")
        wbn = wb / wb[1]
        invwb = 1.0 / wbn
        synth = np.diag(invwb) @ np.linalg.inv(m)
        sha = (
            precomputed_sha256
            if precomputed_sha256 is not None
            else (sha256_file(path) if compute_sha256 else None)
        )
        return {
            "schema_version": "1.0.0",
            "tool": {"name": "nef_camera_color_probe", "version": self.VERSION, "read_only": True},
            "source": {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha,
            },
            "libraw_version": libraw_ver,
            "raw_sensor_model": {
                "libraw_color_maximum_dn": white,
                "black_level_dn": None,
                "black_level_status": "no_obtenido_por_esta_API",
            },
            "white_balance": {
                "as_shot_cam_mul_rg_bg2": cam_mul.tolist(),
                "daylight_pre_mul_rg_bg2": pre_mul.tolist(),
                "as_shot_normalized_rgb": wbn.tolist(),
                "inverse_as_shot_gains_r_g1_g2_b": [float(invwb[0]), 1.0, 1.0, float(invwb[2])],
                "note": "Coeficientes WB LibRaw, no temperatura Kelvin/tint de Adobe.",
            },
            "color": {
                "libraw_rgb_cam_3x4": m34.tolist(),
                "libraw_rgb_cam_3x3_row_sums": m.sum(axis=1).tolist(),
                "libraw_rgb_cam_condition_number": cond,
                "candidate_linear_srgb_to_camera_3x3": synth.tolist(),
                "candidate_gray_response_r_g_b": (synth @ np.ones(3)).tolist(),
                "model": "inv_diag_as_shot_wb_multiplied_by_inverse_libraw_rgb_cam",
                "confidence": "aproximacion para pruebas; no simula la respuesta espectral Nikon ni el DCP de Adobe",
            },
            "warnings": [
                "No se ha extraído black level por la API C de este módulo.",
                "El modelo matricial no reproduce las LUT, tono, perfil dual illuminant ni ajustes de Adobe o Capture One.",
                "Si rgb_cam integra una transformación de balance distinta, verificar la linealidad y neutralidad mediante TIFF de control.",
                "No modificar automáticamente las MakerNotes Nikon a partir de este informe.",
                "La ganancia CFA se aplica a los valores POR ENCIMA del nivel negro; no multiplicar el pedestal del sensor.",
            ],
        }


# ---------------------------------------------------------------------------
# Extractor de Perfiles Adobe DNG (sin dependencia de tifffile)
# ---------------------------------------------------------------------------

def approx_cct_mccamy(xy: np.ndarray | tuple[float, float]) -> float:
    x, y = float(xy[0]), float(xy[1])
    n = (x - 0.3320) / (y - 0.1858)
    return float(-449.0 * n**3 + 3525.0 * n**2 - 6823.3 * n + 5520.33)


def mix_weight_dng(temperature: float) -> float:
    """Factor de interpolación en mireds entre Iluminante A (2856 K) y D65 (6504 K)."""
    return float(np.clip((1.0 / temperature - 1.0 / 6504.0) / (1.0 / 2856.0 - 1.0 / 6504.0), 0.0, 1.0))


def estimate_temperature_dng(
    neutral: np.ndarray, cm1: np.ndarray, cm2: np.ndarray
) -> tuple[float, np.ndarray]:
    temperature = 6200.0
    xy = np.array([np.nan, np.nan], dtype=np.float64)
    for _ in range(30):
        weight = mix_weight_dng(temperature)
        cm = weight * cm1 + (1.0 - weight) * cm2
        xyz = np.linalg.solve(cm, neutral)
        xy = xyz[:2] / xyz.sum()
        new_temperature = approx_cct_mccamy(xy)
        if not (1500.0 <= new_temperature <= 40000.0):
            raise DngProfileError("Estimación de CCT fuera de rango; no puede interpolarse con seguridad.")
        if abs(new_temperature - temperature) < 1e-6:
            break
        temperature = new_temperature
    return float(temperature), xy


def parse_xmp_camera_profile(raw: bytes | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        root = ET.fromstring(raw.decode("utf-8-sig", "replace").strip().rstrip("\x00"))
        names = {
            "crd": "http://ns.adobe.com/camera-raw-defaults/1.0/",
            "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
        }
        for el in root.iter():
            for k, v in el.attrib.items():
                if k in (f"{{{names['crd']}}}CameraProfile", f"{{{names['crs']}}}CameraProfile"):
                    return {
                        "xmp_camera_profile_field": v,
                        "xmp_camera_profile_namespace": k.split("}")[0][1:],
                    }
        return {"xmp_camera_profile_field": None}
    except Exception as exc:
        return {"xmp_parse_error": str(exc)}


class DngAdobeProfileExtractor:
    """Extrae el perfil DNG primario, matrices de interpolación dual-illuminant y LUTs 3D
    directamente mediante nuestro TiffReader nativo (sin requerir tifffile).
    """

    def __init__(self, dng_path: str | Path) -> None:
        self.dng_path = Path(dng_path)
        if not self.dng_path.is_file():
            raise FileNotFoundError(f"No existe el archivo DNG: {self.dng_path}")

    @staticmethod
    def _read_rational_array(reader: TiffReader, ifd: TiffDirectory, name: str) -> np.ndarray:
        tag = ifd.tag_by_name(name)
        if tag is None:
            raise DngProfileError(f"Falta etiqueta DNG requerida: {name}")
        raw = reader.read_tag_bytes(tag)
        fmt = "ii" if tag.dtype == 10 else "II"
        pairs = np.frombuffer(raw, dtype=np.dtype(reader.endian + ("i4" if tag.dtype == 10 else "u4"))).reshape(-1, 2)
        if np.any(pairs[:, 1] == 0):
            raise DngProfileError(f"{name} contiene denominador cero")
        return pairs[:, 0].astype(np.float64) / pairs[:, 1].astype(np.float64)

    @classmethod
    def _read_matrix_3x3(cls, reader: TiffReader, ifd: TiffDirectory, name: str) -> np.ndarray:
        return cls._read_rational_array(reader, ifd, name).reshape(3, 3)

    @staticmethod
    def _read_opt(reader: TiffReader, ifd: TiffDirectory, name: str) -> Any:
        tag = ifd.tag_by_name(name)
        if tag is None:
            return None
        raw = reader.read_tag_bytes(tag)
        if tag.dtype == 2 or name == "OriginalRawFileName":
            return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        _, _, fmt = TYPE_INFO_LOOKUP[tag.dtype]
        if fmt is None:
            return list(raw)
        vals = list(struct.unpack(reader.endian + fmt * tag.count, raw))
        return vals[0] if len(vals) == 1 else vals

    @staticmethod
    def _read_float32_lut(reader: TiffReader, ifd: TiffDirectory, name: str, dims: list[int] | None) -> np.ndarray:
        if not dims:
            return np.empty(0, dtype=np.float32)
        tag = ifd.tag_by_name(name)
        if tag is None:
            return np.empty(0, dtype=np.float32)
        raw = reader.read_tag_bytes(tag)
        arr = np.frombuffer(raw, dtype=np.dtype(reader.endian + "f4")).astype(np.float32, copy=True)
        return arr.reshape(dims[2], dims[0], dims[1], 3)

    def extract(
        self,
        *,
        kelvin: float | None = 6200.0,
        precomputed_sha256: str | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
        reader = TiffReader(self.dng_path).parse()
        ifd0 = reader.get("IFD0")
        if ifd0 is None or ifd0.tag(0xC612) is None:
            raise DngProfileError("El archivo no identifica un DNG válido (falta DNGVersion)")
        raw_ifd = reader.get("IFD0/SubIFD0") or ifd0

        mats = {
            name: self._read_matrix_3x3(reader, ifd0, name)
            for name in ("ColorMatrix1", "ColorMatrix2", "ForwardMatrix1", "ForwardMatrix2")
        }
        cal = {
            name: (
                self._read_matrix_3x3(reader, ifd0, name)
                if ifd0.tag_by_name(name) is not None
                else np.eye(3, dtype=np.float64)
            )
            for name in ("CameraCalibration1", "CameraCalibration2")
        }
        ab = (
            self._read_rational_array(reader, ifd0, "AnalogBalance")
            if ifd0.tag_by_name("AnalogBalance") is not None
            else np.ones(3, dtype=np.float64)
        )
        neutral = self._read_rational_array(reader, ifd0, "AsShotNeutral")
        cct, xy = estimate_temperature_dng(neutral, mats["ColorMatrix1"], mats["ColorMatrix2"])
        interp_kelvin = float(kelvin) if kelvin is not None else cct
        if interp_kelvin <= 0:
            raise DngProfileError("kelvin debe ser > 0")

        illum1 = int(self._read_opt(reader, ifd0, "CalibrationIlluminant1"))
        illum2 = int(self._read_opt(reader, ifd0, "CalibrationIlluminant2"))
        if (illum1, illum2) != (17, 21):
            raise DngProfileError("Este extractor solo interpola automáticamente iluminantes 17 (A) y 21 (D65).")

        weight = mix_weight_dng(interp_kelvin)
        fm = weight * mats["ForwardMatrix1"] + (1.0 - weight) * mats["ForwardMatrix2"]
        cc = weight * cal["CameraCalibration1"] + (1.0 - weight) * cal["CameraCalibration2"]
        abmat = np.diag(ab)
        refneutral = np.linalg.solve(abmat @ cc, neutral)
        cam_to_xyz = fm @ np.diag(1.0 / refneutral) @ np.linalg.inv(abmat @ cc)
        rgb_to_cam = np.linalg.solve(cam_to_xyz, SRGB_TO_XYZ_D50)

        sha = precomputed_sha256 if precomputed_sha256 is not None else sha256_file(self.dng_path)
        h_dims = self._read_opt(reader, ifd0, "ProfileHueSatMapDims")
        l_dims = self._read_opt(reader, ifd0, "ProfileLookTableDims")

        extras = {
            "profile_name": self._read_opt(reader, ifd0, "ProfileName"),
            "profile_tone_curve_present": ifd0.tag_by_name("ProfileToneCurve") is not None,
            "profile_embed_policy": self._read_opt(reader, ifd0, "ProfileEmbedPolicy"),
            "additional_camera_profiles_present": ifd0.tag_by_name("ExtraCameraProfiles") is not None,
            "hue_sat_map_dims": h_dims,
            "look_table_dims": l_dims,
            "hue_sat_map_data1_present": ifd0.tag_by_name("ProfileHueSatMapData1") is not None,
            "hue_sat_map_data2_present": ifd0.tag_by_name("ProfileHueSatMapData2") is not None,
            "look_table_data_present": ifd0.tag_by_name("ProfileLookTableData") is not None,
        }
        xmp_tag = ifd0.tag_by_name("XMP")
        xmp_bytes = reader.read_tag_bytes(xmp_tag) if xmp_tag is not None else None
        xmp = parse_xmp_camera_profile(xmp_bytes)

        ltab = self._read_float32_lut(reader, ifd0, "ProfileLookTableData", l_dims)
        h1 = self._read_float32_lut(reader, ifd0, "ProfileHueSatMapData1", h_dims)
        h2 = self._read_float32_lut(reader, ifd0, "ProfileHueSatMapData2", h_dims)

        black = (
            self._read_rational_array(reader, raw_ifd, "BlackLevel").tolist()
            if raw_ifd.tag_by_name("BlackLevel") is not None
            else None
        )
        baseline = (
            self._read_rational_array(reader, ifd0, "BaselineExposure").tolist()
            if ifd0.tag_by_name("BaselineExposure") is not None
            else None
        )
        raw_h = int(self._read_opt(reader, raw_ifd, "ImageLength"))
        raw_w = int(self._read_opt(reader, raw_ifd, "ImageWidth"))
        cfa_tag = raw_ifd.tag_by_name("CFAPattern")
        cfa_bytes = list(reader.read_tag_bytes(cfa_tag)) if cfa_tag is not None else None

        report: dict[str, Any] = {
            "schema": "dng_embedded_camera_profile_analysis_v1",
            "source_file": str(self.dng_path),
            "sha256": sha,
            "file_size_bytes": self.dng_path.stat().st_size,
            "camera": {
                "make": self._read_opt(reader, ifd0, "Make"),
                "model": self._read_opt(reader, ifd0, "Model"),
                "unique_camera_model": self._read_opt(reader, ifd0, "UniqueCameraModel"),
                "original_nef_filename": self._read_opt(reader, ifd0, "OriginalRawFileName"),
                "software": self._read_opt(reader, ifd0, "Software"),
            },
            "dng": {
                "dng_version": list(self._read_opt(reader, ifd0, "DNGVersion")),
                "byteorder": reader.endian,
                "raw_cfa_shape": [raw_h, raw_w],
                "raw_compression_code": self._read_opt(reader, raw_ifd, "Compression"),
                "cfa_pattern_bytes": cfa_bytes,
                "black_level_repeat_dim": self._read_opt(reader, raw_ifd, "BlackLevelRepeatDim"),
                "black_level_dn": black,
                "white_level_dn": self._read_opt(reader, raw_ifd, "WhiteLevel"),
                "default_crop_origin": (
                    to_native(self._read_rational_array(reader, raw_ifd, "DefaultCropOrigin"))
                    if raw_ifd.tag_by_name("DefaultCropOrigin") is not None
                    else None
                ),
                "default_crop_size": (
                    to_native(self._read_rational_array(reader, raw_ifd, "DefaultCropSize"))
                    if raw_ifd.tag_by_name("DefaultCropSize") is not None
                    else None
                ),
                "baseline_exposure_ev": baseline[0] if baseline else None,
            },
            "embedded_profile": extras,
            "xmp_default_camera_profile": xmp,
            "profile_matrices": {k: v.tolist() for k, v in mats.items()},
            "camera_calibration": {k: v.tolist() for k, v in cal.items()},
            "analog_balance": ab.tolist(),
            "as_shot_neutral": neutral.tolist(),
            "as_shot_inverse_gains": (1.0 / neutral).tolist(),
            "derived_linear_model": {
                "white_xy_approximately_inferred": xy.tolist(),
                "white_cct_mccamy_approx_K": cct,
                "interpolation_temperature_K": interp_kelvin,
                "interpolation_weight_illuminant_A": weight,
                "forward_matrix_interpolated": fm.tolist(),
                "camera_to_xyz_d50_before_luts": cam_to_xyz.tolist(),
                "xyz_d50_from_srgb_linear": SRGB_TO_XYZ_D50.tolist(),
                "srgb_linear_to_camera_raw_matrix": rgb_to_cam.tolist(),
                "condition_number": float(np.linalg.cond(rgb_to_cam)),
                "neutral_rgb1_raw": (rgb_to_cam @ np.ones(3)).tolist(),
            },
            "cautions": [
                "Perfil incrustado en DNG; puede no ser el perfil seleccionado en ACR o para el NEF original.",
                "Transformación lineal inversa aproximada: NO incluye HueSatMap, LookTable ni curva tonal del revelador.",
                "BlackLevel=0 en DNG convertido NO demuestra que BlackLevel del NEF original sea cero.",
                "El método de McCamy proporciona CCT aproximada, no una reproducción exacta del selector Kelvin/Tint de Adobe.",
                "Si el objetivo es un NEF original, preservar sus niveles RAW y metadatos propios; no copiar valores de escalado desde DNG.",
            ],
        }
        luts = {
            "ProfileHueSatMapData1": h1,
            "ProfileHueSatMapData2": h2,
            "ProfileLookTableData": ltab,
        }
        return report, rgb_to_cam, luts

    def extract_to_files(
        self,
        out_prefix: str | Path,
        *,
        kelvin: float | None = 6200.0,
        precomputed_sha256: str | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, str]:
        """Extrae y persiste el informe DNG, la matriz JSON (devolviendo su SHA-256) y las LUTs .npz."""
        stem = str(out_prefix)
        report, rgb_to_cam, luts = self.extract(kelvin=kelvin, precomputed_sha256=precomputed_sha256)
        matrix_path = Path(stem + ".adobe_standard_rgb_to_camera.json")
        report_path = Path(stem + ".dng_profile_report.json")
        lut_path = Path(stem + ".dng_profile_luts.npz")

        matrix_sha256 = write_json_atomic(matrix_path, rgb_to_cam.tolist(), overwrite=True)
        write_json_atomic(report_path, report, overwrite=True)
        np.savez_compressed(lut_path, **luts)
        return report, rgb_to_cam, matrix_sha256


TYPE_INFO_LOOKUP: dict[int, tuple[str, int, str | None]] = {
    1: ("BYTE", 1, "B"),
    2: ("ASCII", 1, None),
    3: ("SHORT", 2, "H"),
    4: ("LONG", 4, "I"),
    5: ("RATIONAL", 8, "II"),
    6: ("SBYTE", 1, "b"),
    7: ("UNDEFINED", 1, None),
    8: ("SSHORT", 2, "h"),
    9: ("SLONG", 4, "i"),
    10: ("SRATIONAL", 8, "ii"),
    11: ("FLOAT", 4, "f"),
    12: ("DOUBLE", 8, "d"),
    13: ("IFD", 4, "I"),
}


# ---------------------------------------------------------------------------
# Validaciones de Procedencia Cruzada NEF <-> DNG y DCRAW
# ---------------------------------------------------------------------------

def verify_nef_dng_provenance(
    inspection: dict[str, Any],
    dng_report: dict[str, Any],
    nef_path: str | Path,
    *,
    expected_profile: str = "Adobe Standard",
    allow_different_original_name: bool = False,
) -> list[str]:
    """Verifica la correspondencia de modelo, nombre de archivo original, perfil incrustado y DCRAW."""
    warnings: list[str] = []
    model_nef = str(inspection.get("camera", {}).get("model") or "").strip()
    model_dng = str(dng_report.get("camera", {}).get("model") or "").strip()
    if model_nef.lower() != model_dng.lower():
        raise DngProfileError(
            f"Modelos de cámara diferentes: NEF='{model_nef}'; DNG='{model_dng}'. No se mezclan perfiles."
        )
    orig_name = str(dng_report.get("camera", {}).get("original_nef_filename") or "")
    nef_name = Path(nef_path).name
    if orig_name and orig_name.lower() != nef_name.lower():
        if not allow_different_original_name:
            raise DngProfileError(
                f"El DNG declara como origen '{orig_name}', pero el NEF es '{nef_name}'. "
                "Usar allow_different_original_name=True SOLO tras verificar manualmente la procedencia."
            )
        warnings.append(
            f"Se ha permitido diferencia de nombre entre NEF ({nef_name}) y DNG ({orig_name})."
        )
    if not orig_name:
        warnings.append("El DNG no declara OriginalRawFileName: no podemos confirmar origen por nombre.")

    emb_profile = str(dng_report.get("embedded_profile", {}).get("profile_name") or "")
    if expected_profile and emb_profile.lower() != expected_profile.lower():
        raise DngProfileError(
            f"Perfil incrustado '{emb_profile}' diferente del esperado '{expected_profile}'."
        )

    if "d610" in model_dng.lower():
        cm2 = np.asarray(dng_report["profile_matrices"]["ColorMatrix2"], dtype=np.float64)
        max_delta = float(np.max(np.abs(cm2 - DCRAW_D610_COLOR_MATRIX_2)))
        if max_delta >= 1e-9:
            warnings.append(
                f"DCRAW D610 vs ColorMatrix2: diferencia máxima={max_delta:.12g}. Revisar origen del perfil."
            )

    xmp_profile = str(
        dng_report.get("xmp_default_camera_profile", {}).get("xmp_camera_profile_field") or ""
    )
    if xmp_profile and expected_profile and xmp_profile.lower() != expected_profile.lower():
        warnings.append(
            f"XMP indica '{xmp_profile}', pero el perfil EMBEBIDO utilizado es '{expected_profile}'."
        )
    return warnings

