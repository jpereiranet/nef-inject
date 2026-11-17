"""Simulador optoelectrónico de sensor post-CFA y caracterizador empírico (darks/flats).

Forma parte de la librería `nefinject`, modelando la física de conversión fotón-electrón,
las fuentes de ruido analógico/digital y las imperfecciones espaciales del sensor semiconductor
directamente sobre el mosaico Bayer crudo en formato uint16.

Modelos físicos integrados (`NefSensorSimulator`, `SensorConfig`):
1. Ruido de Patrón Fijo (FPN):
   - DSNU (Dark Signal Non-Uniformity): Señal parásita aditiva por corriente oscura en sensels.
   - PRNU (Photo Response Non-Uniformity): Modulación espacial multiplicativa de ganancia sensel a sensel.
2. Defectos de Sensels (`Defects`):
   - Píxeles muertos (bloqueados en nivel de negro) y calientes (saturados o con corriente oscura anómala).
3. Ruido Temporal:
   - Ruido de disparo fotónico (Poisson): Fluctuación estadística cuántica del flujo de fotones incidente.
   - Ruido de lectura (Gaussiano): Ruido térmico de amplificación Johnson-Nyquist y ruido de conversión ADC.
4. Bandas de Lectura (Row/Column Banding):
   - Variaciones periódicas o estocásticas inducidas por los circuitos de amplificación por columnas.
5. Sombreado de Sensibilidad de Sensor (`Vignette`):
   - Pérdida de eficiencia cuántica angular en las esquinas por la geometría de micro-lentes integradas.

Arquitectura y Caracterización:
- Generación Pseudoaleatoria Determinista: Arquitectura de RNG por franjas horizontales (`tile_rows`)
  con generadores desacoplados para mapas fijos (`fixed_seed`) y ruido temporal (`temporal_seed`).
- Caracterizador Empírico (`characterize_sensor`):
  Extracción automática de mapas FPN (DSNU y PRNU) y perfiles estadísticos a partir de tomas
  reales o simuladas de fotogramas oscuros (dark frames) y fotogramas uniformes (flat frames).

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .models import (
    NefInjectError,
    save_npy_and_sha256,
    sha256_file,
    sha256_npy_in_memory,
    write_json_atomic,
)
from .optics import _gaussian_1d_numpy

VERSION = "1.0.0"
CFA_OPTIONS = ("RGGB", "BGGR", "GRBG", "GBRG")
CHS = ("R", "G1", "G2", "B")


class SensorSimulationError(NefInjectError):
    """Error de validación o ejecución en el simulador optoelectrónico de sensor."""


@dataclass(frozen=True)
class Vignette:
    edge_factor: float = 1.0  # factor en esquinas (radio normalizado=1)
    power: float = 2.0

    def validate(self) -> None:
        if not (0.0 < self.edge_factor <= 1.0):
            raise ValueError("vignette.edge_factor debe estar en (0,1]")
        if not (self.power > 0 and math.isfinite(self.power)):
            raise ValueError("vignette.power debe ser positivo y finito")


@dataclass(frozen=True)
class Defects:
    dead_ppm: float = 0.0
    hot_ppm: float = 0.0
    hot_e: float = 0.0

    def validate(self) -> None:
        if self.dead_ppm < 0 or self.hot_ppm < 0 or (self.dead_ppm + self.hot_ppm > 1_000_000):
            raise ValueError("Fracciones de defectos ppm invalidas")
        if not (math.isfinite(self.hot_e) and self.hot_e >= 0):
            raise ValueError("hot_e invalido")


@dataclass(frozen=True)
class FixedMaps:
    """Archivos .npy float32/float64 con geometría 1:1 con el CFA del NEF.
    - prnu_npy: factor multiplicativo 1+delta (no delta).
    - dsnu_e_npy: desplazamiento aditivo en electrones.
    - defects_npy: 0 normal, 1 muerto, 2 caliente.
    """

    prnu_npy: str | None = None
    dsnu_e_npy: str | None = None
    defects_npy: str | None = None

    def validate(self) -> None:
        for key in ("prnu_npy", "dsnu_e_npy", "defects_npy"):
            f = getattr(self, key)
            if f is not None and (not isinstance(f, str) or not Path(f).is_file()):
                raise ValueError(f"Mapa fijo inexistente: {key}={f}")


@dataclass(frozen=True)
class SensorConfig:
    cfa: str = "RGGB"
    black_dn: int = 0
    white_dn: int = 15000
    electrons_per_dn: float = 4.0  # Valor ilustrativo salvo medida PTC
    shot_noise_enabled: bool = True
    read_noise_e: float = 0.0
    prnu_fraction: float = 0.0
    dsnu_e: float = 0.0
    row_fpn_e: float = 0.0
    col_fpn_e: float = 0.0
    row_temporal_e: float = 0.0
    col_temporal_e: float = 0.0
    dark_current_e_per_s: float = 0.0
    exposure_s: float = 0.0
    vignette: Vignette = field(default_factory=Vignette)
    color_shading_edge_factors: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    defects: Defects = field(default_factory=Defects)
    fixed_maps: FixedMaps = field(default_factory=FixedMaps)
    fixed_seed: int = 1731
    temporal_seed: int = 1732
    tile_rows: int = 128

    @classmethod
    def from_json(cls, path: str | Path) -> "SensorConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("La configuracion JSON debe ser un objeto")
        fields = set(cls.__dataclass_fields__)
        unknown = set(d) - fields - {"schema_version", "comment", "comments"}
        if unknown:
            raise ValueError(f"Campos desconocidos: {sorted(unknown)}")
        basic = {
            k: v
            for k, v in d.items()
            if k in fields
            and k not in ("vignette", "defects", "fixed_maps", "color_shading_edge_factors")
        }
        vf = Vignette(**d.get("vignette", {}))
        df = Defects(**d.get("defects", {}))
        cf = d.get("color_shading_edge_factors", [1, 1, 1, 1])
        if isinstance(cf, dict):
            cf = [cf[x] for x in CHS]
        if len(cf) != 4:
            raise ValueError("color_shading_edge_factors requiere R,G1,G2,B")
        md = d.get("fixed_maps", {})
        if not isinstance(md, dict):
            raise ValueError("fixed_maps debe ser objeto")
        basedir = Path(path).resolve().parent
        map_kwargs = {
            key: (str((basedir / value).resolve()) if value else None) for key, value in md.items()
        }
        mf = FixedMaps(**map_kwargs)
        conf = cls(
            **basic,
            vignette=vf,
            defects=df,
            fixed_maps=mf,
            color_shading_edge_factors=tuple(map(float, cf)),  # type: ignore[arg-type]
        )
        conf.validate()
        return conf

    @classmethod
    def load(cls, spec: "SensorConfig | str | Path") -> tuple["SensorConfig", str]:
        """Carga un ``SensorConfig`` desde una instancia o una ruta explícita a JSON."""
        if isinstance(spec, cls):
            spec.validate()
            return spec, "in_memory_dataclass"
        p = Path(str(spec))
        if p.suffix.lower() == ".json" and p.is_file():
            return cls.from_json(p), str(p.resolve())
        raise FileNotFoundError(
            f"No se encontró el perfil de sensor JSON '{spec}'. "
            "Indica una ruta explícita a un archivo .json existente."
        )

    def with_levels(self, *, black_dn: int, white_dn: int, cfa: str | None = None) -> "SensorConfig":
        """Devuelve una copia del SensorConfig sincronizando black_dn, white_dn y opcionalmente cfa."""
        from dataclasses import replace

        return replace(
            self,
            black_dn=int(black_dn),
            white_dn=int(white_dn),
            cfa=(cfa.upper() if cfa is not None else self.cfa),
        )

    def validate(self) -> None:
        if self.cfa not in CFA_OPTIONS:
            raise ValueError("CFA Bayer desconocido")
        if not (0 <= self.black_dn < self.white_dn <= 65535):
            raise ValueError("Niveles de negro/blanco invalidos")
        if not math.isfinite(self.electrons_per_dn) or self.electrons_per_dn <= 0:
            raise ValueError("electrons_per_dn debe ser positivo")
        for key in (
            "read_noise_e",
            "prnu_fraction",
            "dsnu_e",
            "row_fpn_e",
            "col_fpn_e",
            "row_temporal_e",
            "col_temporal_e",
            "dark_current_e_per_s",
            "exposure_s",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key}: debe ser >=0, finito")
        if self.prnu_fraction > 0.5:
            raise ValueError("prnu_fraction >0.5 no admite interpretacion gaussiana sencilla")
        for f in self.color_shading_edge_factors:
            if not math.isfinite(f) or not 0 < f <= 2.0:
                raise ValueError("Factor de color shading fuera de (0,2]")
        if not (2 <= self.tile_rows <= 8192):
            raise ValueError("tile_rows fuera de [2,8192]")
        for name in ("fixed_seed", "temporal_seed"):
            if not (0 <= getattr(self, name) < (2**63)):
                raise ValueError(f"{name} fuera del intervalo")
        self.vignette.validate()
        self.defects.validate()
        self.fixed_maps.validate()


class NefSensorSimulator:
    """Simulador de ruido y artefactos optoelectrónicos sobre mosaicos CFA Bayer uint16."""

    def __init__(self, config: SensorConfig) -> None:
        config.validate()
        self.c = config

    def _channel_tile(self, y0: int, y1: int, width: int) -> np.ndarray:
        pattern = self.c.cfa
        pairs = [
            0 if v == "R" else (3 if v == "B" else (1 if pattern[:i].count("G") == 0 else 2))
            for i, v in enumerate(pattern)
        ]
        tile = np.empty((y1 - y0, width), dtype=np.uint8)
        for y in (0, 1):
            for x in (0, 1):
                first = (y - y0) % 2
                tile[first::2, x::2] = pairs[y * 2 + x]
        return tile

    def simulate_in_memory(
        self,
        src_bayer: np.ndarray,
        *,
        source_meta: dict[str, Any] | None = None,
        source_sha256: str | None = None,
        source_filename: str = "bayer_final.npy",
        source_manifest_path: str | None = None,
        output_filename: str = "bayer_sensor.npy",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Ejecuta la simulación de sensor sobre un array 2D uint16 en RAM manteniendo
        reproducibilidad determinista garantizada (orden idéntico de RNG por tile_rows).
        Devuelve (bayer_degradado_u16, report_dict).
        """
        src = np.asarray(src_bayer)
        if src.dtype != np.uint16 or src.ndim != 2:
            raise ValueError("Se requiere RAW Bayer 2D uint16")
        H, W = map(int, src.shape)
        if H % 2 or W % 2:
            raise ValueError("El mosaico CFA requiere dimensiones pares")
        if int(src.max()) > self.c.white_dn:
            raise ValueError("La entrada contiene DN mayores que white_dn; revisar configuracion")

        map_paths = {k: getattr(self.c.fixed_maps, k) for k in ("prnu_npy", "dsnu_e_npy", "defects_npy")}
        fixed_arrays: dict[str, np.ndarray] = {}
        for name, fname in map_paths.items():
            if not fname:
                continue
            data = np.load(fname, mmap_mode="r", allow_pickle=False)
            if data.ndim != 2 or data.shape != (H, W):
                raise ValueError(f"Mapa fijo {name}: resolucion {data.shape} no coincide con {(H, W)}")
            if name == "defects_npy":
                if data.dtype.kind not in "iu" or np.any((data != 0) & (data != 1) & (data != 2)):
                    raise ValueError("defects_npy necesita mascara 0/1/2 entera")
            elif data.dtype.kind != "f" or not np.isfinite(data).all():
                raise ValueError(f"Mapa {name}: requiere float finito")
            if name == "prnu_npy" and (data.min() < 0 or data.max() > 10):
                raise ValueError("PRNU map factor fuera de [0,10]")
            fixed_arrays[name] = data

        h_in = source_sha256 if source_sha256 is not None else sha256_npy_in_memory(src)
        if source_meta is not None:
            orig = source_meta.get("array", {})
            if orig.get("sha256") and orig["sha256"].lower() != h_in.lower():
                raise ValueError("SHA256 entrada no coincide con su manifiesto")
            tgt = source_meta.get("target", {})
            if tgt.get("cfa") and tgt["cfa"].upper() != self.c.cfa:
                raise ValueError("CFA del manifiesto diferente de configuracion")
            proc = source_meta.get("processing", {})
            if ("black_dn" in proc and int(proc["black_dn"]) != self.c.black_dn) or (
                "white_dn" in proc and int(proc["white_dn"]) != self.c.white_dn
            ):
                raise ValueError("black_dn/white_dn no coinciden con manifiesto de origen")

        c = self.c
        fixed = np.random.default_rng(c.fixed_seed)
        temporal = np.random.default_rng(c.temporal_seed)
        row_fixed = fixed.normal(0, c.row_fpn_e, size=H) if c.row_fpn_e else np.zeros(H)
        col_fixed = fixed.normal(0, c.col_fpn_e, size=W) if c.col_fpn_e else np.zeros(W)
        row_temp = temporal.normal(0, c.row_temporal_e, size=H) if c.row_temporal_e else np.zeros(H)
        col_temp = temporal.normal(0, c.col_temporal_e, size=W) if c.col_temporal_e else np.zeros(W)
        xx = np.arange(W, dtype=np.float32)[None, :]
        nx = (xx - (W - 1) / 2.0) / max((W - 1) / 2.0, 1.0)
        norm = np.float32(2.0)
        dtype = np.float64 if c.electrons_per_dn >= 1e7 else np.float32

        arr = np.empty((H, W), dtype=np.uint16)
        min_out = 65535
        max_out = 0
        num_changed = 0
        num_zero_clip = 0
        num_white_clip = 0

        for y0 in range(0, H, c.tile_rows):
            y1 = min(H, y0 + c.tile_rows)
            raw = np.asarray(src[y0:y1, :], dtype=dtype)
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            ny = (yy - (H - 1) / 2.0) / max((H - 1) / 2.0, 1.0)
            r2 = np.minimum(1.0, (nx * nx + ny * ny) / norm)
            r = np.sqrt(r2)
            ch = self._channel_tile(y0, y1, W)
            v = 1.0 - (1.0 - c.vignette.edge_factor) * np.power(r, c.vignette.power)
            if any(x != 1 for x in c.color_shading_edge_factors):
                shading = np.take(np.array(c.color_shading_edge_factors, dtype=np.float32), ch)
                v = v * (1.0 + (shading - 1.0) * np.power(r, c.vignette.power))
            e = np.maximum(raw - c.black_dn, 0.0) * c.electrons_per_dn * v
            if "prnu_npy" in fixed_arrays:
                e *= np.asarray(fixed_arrays["prnu_npy"][y0:y1, :], dtype=dtype)
            if c.prnu_fraction > 0:
                e *= np.maximum(0.0, 1.0 + fixed.normal(0, c.prnu_fraction, size=e.shape))
            if c.shot_noise_enabled:
                e = temporal.poisson(e).astype(dtype, copy=False)
            if c.dark_current_e_per_s and c.exposure_s:
                e += temporal.poisson(c.dark_current_e_per_s * c.exposure_s, size=e.shape)
            if "dsnu_e_npy" in fixed_arrays:
                e += np.asarray(fixed_arrays["dsnu_e_npy"][y0:y1, :], dtype=dtype)
            if c.dsnu_e:
                e += fixed.normal(0, c.dsnu_e, size=e.shape)
            if c.row_fpn_e:
                e += row_fixed[y0:y1, None]
            if c.col_fpn_e:
                e += col_fixed[None, :]
            if c.read_noise_e:
                e += temporal.normal(0, c.read_noise_e, size=e.shape)
            if c.row_temporal_e:
                e += row_temp[y0:y1, None]
            if c.col_temporal_e:
                e += col_temp[None, :]
            if c.defects.dead_ppm or c.defects.hot_ppm:
                draw = fixed.random(size=e.shape)
                if c.defects.dead_ppm:
                    e[draw < c.defects.dead_ppm / 1e6] = 0.0
                if c.defects.hot_ppm:
                    mask_hot = (draw >= c.defects.dead_ppm / 1e6) & (
                        draw < (c.defects.dead_ppm + c.defects.hot_ppm) / 1e6
                    )
                    e[mask_hot] += c.defects.hot_e
            if "defects_npy" in fixed_arrays:
                m = np.asarray(fixed_arrays["defects_npy"][y0:y1, :])
                e[m == 1] = 0.0
                e[m == 2] += c.defects.hot_e
            dn = c.black_dn + e / c.electrons_per_dn
            num_zero_clip += int(np.count_nonzero(dn < c.black_dn))
            num_white_clip += int(np.count_nonzero(dn > c.white_dn))
            q = np.rint(np.clip(dn, c.black_dn, c.white_dn)).astype(np.uint16)
            arr[y0:y1, :] = q
            num_changed += int(np.count_nonzero(q != raw))
            min_out = min(min_out, int(q.min()))
            max_out = max(max_out, int(q.max()))

        out_sha = sha256_npy_in_memory(arr)
        report: dict[str, Any] = {
            "schema_version": "1.0.0",
            "tool": {"name": "nefinject.sensor", "version": VERSION},
            "type": "synthetic_cfa_raw_sensor_perturbation",
            "source": {
                "filename": source_filename,
                "sha256": h_in,
                "manifest": source_manifest_path,
                "reference_nef_sha256": (
                    source_meta.get("target", {}).get("reference_nef_sha256")
                    if source_meta
                    else None
                ),
            },
            "target": {
                **((source_meta or {}).get("target", {})),
                "cfa": c.cfa,
                "width": W,
                "height": H,
                "reference_nef_sha256": (
                    source_meta.get("target", {}).get("reference_nef_sha256")
                    if source_meta
                    else None
                ),
            },
            "processing": {
                **((source_meta or {}).get("processing", {})),
                "source_black_dn": c.black_dn,
                "black_dn": c.black_dn,
                "source_white_dn": c.white_dn,
                "white_dn": c.white_dn,
                "sensor_simulator": asdict(c),
                "fixed_maps_sha256": {
                    k: sha256_file(Path(v)) for k, v in map_paths.items() if v
                },
                "processing_order": [
                    "linear CFA DN->approx electrons",
                    "vignetting+color shading",
                    "PRNU",
                    "Poisson shot (configurable)",
                    "Poisson dark",
                    "DSNU + fixed row/col",
                    "temporal read + row/col",
                    "defects",
                    "clip and quantize",
                ],
            },
            "array": {
                "filename": output_filename,
                "sha256": out_sha,
                "shape_hw": [H, W],
                "dtype": "uint16",
                "min_dn": min_out,
                "max_dn": max_out,
            },
            "quality": {
                "changed_photosites": num_changed,
                "changed_fraction": num_changed / (H * W),
                "samples_below_black_before_clipping": num_zero_clip,
                "samples_above_white_before_clipping": num_white_clip,
            },
            "limitations": [
                "Los parametros son ilustrativos hasta estimarse por ISO/tiempo de exposicion.",
                "El input uint16 ya esta cuantificado: no recupera microdetalles eliminados en la sintesis anterior.",
                "PSF/OLPF debe aplicarse antes del muestreo CFA.",
                "El nivel negro de cero recorta la mitad negativa del ruido de sombras si este es apreciable.",
                "Los patrones fijos se repiten para igual geometria, config y fixed_seed; si cambian componentes RNG activos, puede cambiar su realizacion.",
            ],
        }
        return arr, report

    def simulate(
        self,
        src_path: str | Path,
        output_path: str | Path,
        *,
        source_manifest: str | Path | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Ejecuta la simulación de sensor persistiendo entradas y salidas en disco."""
        src_p = Path(src_path)
        out_p = Path(output_path)
        if src_p.suffix.lower() != ".npy" or out_p.suffix.lower() != ".npy":
            raise ValueError("Entrada y salida deben terminar en .npy")
        if not src_p.is_file():
            raise FileNotFoundError(src_p)
        if src_p.resolve() == out_p.resolve():
            raise ValueError("La salida no puede sobrescribir la referencia de entrada")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        manifest_out = out_p.with_suffix(".json")
        if not overwrite and (out_p.exists() or manifest_out.exists()):
            raise FileExistsError("La salida ya existe: utilizar otro nombre de experimento")

        src_arr = np.load(src_p, mmap_mode="r", allow_pickle=False)
        h_in = sha256_file(src_p)
        source_meta = None
        if source_manifest is None and src_p.with_suffix(".json").is_file():
            source_manifest = src_p.with_suffix(".json")
        if source_manifest is not None:
            source_manifest = Path(source_manifest)
            source_meta = json.loads(source_manifest.read_text(encoding="utf-8"))

        arr, report = self.simulate_in_memory(
            src_arr,
            source_meta=source_meta,
            source_sha256=h_in,
            source_filename=src_p.name,
            source_manifest_path=str(source_manifest) if source_manifest else None,
            output_filename=out_p.name,
        )
        save_npy_and_sha256(out_p, arr, overwrite=overwrite)
        try:
            write_json_atomic(manifest_out, report, overwrite=overwrite)
        except Exception:
            out_p.unlink(missing_ok=True)
            raise
        return report


# ---------------------------------------------------------------------------
# Caracterizador empírico de sensor (dark frames + flat fields -> PRNU, DSNU, PTC)
# ---------------------------------------------------------------------------

def _read_many(paths: Sequence[Path]) -> list[np.ndarray]:
    if len(paths) < 2:
        raise ValueError("Se necesitan al menos dos capturas de cada tipo")
    files = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
    shp = files[0].shape
    if len(shp) != 2 or shp[0] % 2 or shp[1] % 2:
        raise ValueError("RAW CFA debe ser HxW con dimensiones pares")
    for i, a in enumerate(files):
        if a.shape != shp or a.dtype != np.uint16:
            raise ValueError(f"Archivo {paths[i]} incompatible: forma o dtype")
    return files


def _channel_slices(cfa: str):
    if len(cfa) != 4 or sorted(cfa) != sorted("RGGB"):
        raise ValueError("CFA invalido")
    names = ["R", "G1", "G2", "B"]
    gs = 0
    for i, x in enumerate(cfa):
        if x == "G":
            name = names[1 + gs]
            gs += 1
        else:
            name = x
        yield name, (slice(i // 2, None, 2), slice(i % 2, None, 2))


def characterize_sensor(
    darks: Sequence[str | Path],
    flats: Sequence[str | Path],
    prefix: str | Path,
    *,
    cfa: str = "RGGB",
    blur_sigma_px: float = 35.0,
    black_dn: int = 0,
    white_dn: int = 15000,
    assumed_e_per_dn: float | None = None,
) -> dict[str, Any]:
    """Estima ganancia PTC (e-/DN), mapas de PRNU, DSNU y defectos a partir de >=2 darks y >=2 flats."""
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    if blur_sigma_px < 3 or not math.isfinite(blur_sigma_px):
        raise ValueError("Se requiere blur_sigma_px >=3")
    if not 0 <= black_dn < white_dn <= 65535:
        raise ValueError("Rango DN invalido")
    if assumed_e_per_dn is not None and not (
        math.isfinite(assumed_e_per_dn) and assumed_e_per_dn > 0
    ):
        raise ValueError("electrons-per-dn invalido")

    prefix_p = Path(prefix)
    dpaths = [Path(p) for p in darks]
    fpaths = [Path(p) for p in flats]
    dfiles = _read_many(dpaths)
    ffiles = _read_many(fpaths)
    if dfiles[0].shape != ffiles[0].shape:
        raise ValueError("Flat y dark difieren en resolucion")
    h, w = dfiles[0].shape
    prefix_p.parent.mkdir(parents=True, exist_ok=True)
    f_prnu = prefix_p.with_name(prefix_p.name + "_prnu.npy")
    f_dsnu = prefix_p.with_name(prefix_p.name + "_dsnu_e.npy")
    f_mask = prefix_p.with_name(prefix_p.name + "_defects.npy")
    f_profile = prefix_p.with_name(prefix_p.name + "_sensor_profile.json")
    f_report = prefix_p.with_name(prefix_p.name + "_characterization.json")
    for p in (f_prnu, f_dsnu, f_mask, f_profile, f_report):
        if p.exists():
            raise FileExistsError(f"Salida existente: {p}")

    dm = np.zeros((h, w), dtype=np.float32)
    fm = np.zeros((h, w), dtype=np.float32)
    for a in dfiles:
        dm += np.asarray(a, dtype=np.float32) / len(dfiles)
    for a in ffiles:
        fm += np.asarray(a, dtype=np.float32) / len(ffiles)
    signal = np.maximum(fm - dm, 0.0)

    per_channel: dict[str, Any] = {}
    all_gain: list[float] = []
    all_dark_var: list[float] = []
    for name, sl in _channel_slices(cfa):
        roi = signal[sl]
        valid = (roi > 30) & (fm[sl] < white_dn * 0.9)
        if int(valid.sum()) < 40:
            per_channel[name] = {"status": "insufficient_valid_flat_pixels"}
            continue
        df = np.asarray(dfiles[0][sl], dtype=np.float32) - np.asarray(
            dfiles[1][sl], dtype=np.float32
        )
        ff = np.asarray(ffiles[0][sl], dtype=np.float32) - np.asarray(
            ffiles[1][sl], dtype=np.float32
        )
        diff_d = df[valid]
        diff_f = ff[valid]
        dark_var = float(np.var(diff_d - diff_d.mean()) * 0.5)
        flat_var = float(np.var(diff_f - diff_f.mean()) * 0.5)
        mean_signal = float(roi[valid].mean())
        shot_var = max(flat_var - dark_var, 0.0)
        gain = mean_signal / shot_var if shot_var > 1e-8 else None
        per_channel[name] = {
            "mean_signal_dn": mean_signal,
            "dark_temporal_var_dn2": dark_var,
            "flat_temporal_var_dn2": flat_var,
            "shot_var_dn2_estimate": shot_var,
            "electrons_per_dn_estimate": gain,
            "n_valid": int(valid.sum()),
        }
        if gain is not None and 0.1 < gain < 1e5:
            all_gain.append(gain)
        all_dark_var.append(dark_var)

    if assumed_e_per_dn is None:
        if not all_gain:
            raise ValueError("No se ha podido estimar e-/DN: usar --electrons-per-dn")
        e_per_dn = float(np.median(all_gain))
    else:
        e_per_dn = assumed_e_per_dn

    prnu = np.ones((h, w), dtype=np.float32)
    for _, sl in _channel_slices(cfa):
        sub = np.ascontiguousarray(signal[sl], dtype=np.float32)
        if cv2 is not None:
            smooth = cv2.GaussianBlur(
                sub,
                (0, 0),
                sigmaX=blur_sigma_px / 2.0,
                sigmaY=blur_sigma_px / 2.0,
                borderType=cv2.BORDER_REFLECT_101,
            )
        else:
            smooth = _gaussian_1d_numpy(
                _gaussian_1d_numpy(sub, blur_sigma_px / 2.0, axis=1),
                blur_sigma_px / 2.0,
                axis=0,
            )
        valid = smooth > np.maximum(20.0, np.quantile(smooth, 0.05) * 0.5)
        p = np.ones_like(sub)
        np.divide(sub, smooth, out=p, where=valid)
        prnu[sl] = np.clip(p, 0.0, 10.0)

    dsnu = (dm - np.median(dm)).astype(np.float32) * np.float32(e_per_dn)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[prnu < 0.5] = 1
    mask[(prnu > 1.5) & (mask == 0)] = 2
    mask[(dm - np.median(dm) > 100) & (mask == 0)] = 2
    np.save(f_prnu, prnu)
    np.save(f_dsnu, dsnu)
    np.save(f_mask, mask)

    read_e = float(np.sqrt(np.median(all_dark_var)) * e_per_dn) if all_dark_var else 0.0
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "tool": "nefinject.sensor",
        "version": VERSION,
        "inputs": {
            "dark": [{"file": str(x), "sha256": sha256_file(x)} for x in dpaths],
            "flat": [{"file": str(x), "sha256": sha256_file(x)} for x in fpaths],
        },
        "geometry": {"height": h, "width": w, "cfa": cfa},
        "method": {
            "blurring_sigma_px": blur_sigma_px,
            "black_dn": black_dn,
            "white_dn": white_dn,
            "ptc_method": "two-frame difference variance, one flat illumination, approximate",
            "dark_offset_reference": "global median of average dark frame",
        },
        "estimates": {
            "electrons_per_dn": e_per_dn,
            "read_noise_e_includes_dark_temporal": read_e,
            "per_channel": per_channel,
            "prnu_std": float(np.std(prnu - 1)),
            "dsnu_std_e": float(dsnu.std()),
            "dead_pixels": int(np.sum(mask == 1)),
            "hot_pixels": int(np.sum(mask == 2)),
        },
        "maps": {
            "prnu_npy": f_prnu.name,
            "dsnu_e_npy": f_dsnu.name,
            "defects_npy": f_mask.name,
        },
        "limitations": [
            "Valores solo validos para ISO y condiciones de captura concretas.",
            "El PTC con un solo nivel de flat NO sustituye regresion con multiples niveles de irradiacion.",
            "La estabilidad de iluminacion y el preprocesado NEF pueden sesgar e-/DN.",
            "El criterio de pixeles defectuosos es heuristico; revisar antes de uso metrologico.",
            "La separacion optica/PRNU requiere flat-fields calibrados.",
        ],
    }
    profile: dict[str, Any] = {
        "schema_version": "1.0.0",
        "comment": "PERFIL ESTIMADO, validar condicion ISO/exposicion y report. Mapas por fotosito.",
        "cfa": cfa,
        "black_dn": black_dn,
        "white_dn": white_dn,
        "electrons_per_dn": e_per_dn,
        "shot_noise_enabled": True,
        "read_noise_e": read_e,
        "prnu_fraction": 0.0,
        "dsnu_e": 0.0,
        "row_fpn_e": 0.0,
        "col_fpn_e": 0.0,
        "row_temporal_e": 0.0,
        "col_temporal_e": 0.0,
        "dark_current_e_per_s": 0.0,
        "exposure_s": 0.0,
        "vignette": {"edge_factor": 1.0, "power": 2.0},
        "color_shading_edge_factors": [1, 1, 1, 1],
        "defects": {"dead_ppm": 0.0, "hot_ppm": 0.0, "hot_e": 5000.0},
        "fixed_maps": {
            "prnu_npy": f_prnu.name,
            "dsnu_e_npy": f_dsnu.name,
            "defects_npy": f_mask.name,
        },
        "fixed_seed": 1731,
        "temporal_seed": 1732,
        "tile_rows": 128,
    }
    try:
        write_json_atomic(f_profile, profile, overwrite=False)
        write_json_atomic(f_report, report, overwrite=False)
    except Exception:
        for p in (f_prnu, f_dsnu, f_mask, f_profile, f_report):
            p.unlink(missing_ok=True)
        raise
    return report


# Alias funcional
characterize = characterize_sensor

