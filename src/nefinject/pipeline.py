"""Orquestador integral en memoria del pipeline de inyección y simulación RAW Nikon NEF.

Forma parte de la librería `nefinject`, coordinando de extremo a extremo todas las fases
del flujo de trabajo: decodificación del archivo plantilla, extracción de matrices y perfiles,
simulación óptica, síntesis de mosaico Bayer, simulación de sensor, compresión Huffman sin pérdida,
parcheo atómico del contenedor TIFF-EP y generación de manifiestos criptográficos.

Flujo de ejecución de `inject_synthetic_nef` y `NefSession`:
1. Fase de Preparación y Metadatos:
   - Lectura estructural del contenedor NEF plantilla mediante `TiffReader` y `NikonMakerNoteInspector`.
   - Extracción de perfiles de color desde el archivo Adobe DNG de referencia (`DngAdobeProfileExtractor`).
   - Verificación estricta de procedencia cruzada (NEF/DNG) y cálculo de sumas SHA-256 iniciales.

2. Fase de Simulación Óptica Pre-CFA (`nefinject.optics`):
   - Carga y linealización electro-óptica de la carta o imagen de entrada.
   - Aplicación en memoria de aberraciones, dispersión de punto (PSF), filtro antialiasing (OLPF),
     viñeteado óptico y distorsión si se definen en la configuración.

3. Fase de Síntesis Bayer CFA y Color (`nefinject.synthesis`):
   - Adaptación de temperatura correlacionada (CCT) e interpolación de matrices de color del perfil DNG.
   - Proyección al espacio de color del sensor, modulación por canales y cuantización a 14 bits.

4. Fase de Perturbación Optoelectrónica de Sensor (`nefinject.sensor`):
   - Inyección de mapas de no uniformidad (DSNU, PRNU), defectos puntuales y ruidos estocásticos (Poisson/Gauss).

5. Fase de Codificación y Parcheo Transaccional (`nefinject.codec` y `nefinject.container`):
   - Compresión Nikon 14-bit Lossless acelerada (JIT Numba o NumPy vectorizado).
   - Sustitución atómica del flujo binario en el contenedor NEF actualizando los punteros TIFF.

6. Fase de Validación y Trazabilidad (`nefinject.calibration`):
   - Generación del informe consolidado `PipelineResult`, cálculo de huellas SHA-256 de todos
     los artefactos y persistencia transaccional del resumen JSON del experimento.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .calibration import CameraColorProfile, NefColorCalibrator, compare_bayer_syntheses
from .codec import NikonLosslessCodec, numba_acceleration_available
from .color import (
    DngAdobeProfileExtractor,
    NefCameraColorProbe,
    verify_nef_dng_provenance,
)
from .container import (
    LibRawNativeProbe,
    NefContainerPatcher,
    NefInspector,
    NefLosslessContext,
    NikonMakerNoteInspector,
    RawPyBackend,
    TiffReader,
)
from .models import (
    InjectionConfig,
    InjectionError,
    PayloadDecodeError,
    PipelineResult,
    SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    read_json,
    save_npy_and_sha256,
    sha256_bytes,
    sha256_file,
    sha256_npy_in_memory,
    write_json_atomic,
)
from .optics import OpticsConfig, simulate_optics, simulate_optics_in_memory
from .sensor import NefSensorSimulator, SensorConfig
from .synthesis import (
    BayerConversionConfig,
    BayerTarget,
    ImageToBayerConverter,
    generate_colorchecker24_png,
    generate_siemens_star_png,
    make_grid_layout,
    prepare_preview_rgb8,
)


class NefSession:
    """Sesión persistente en memoria sobre un archivo plantilla .NEF (y opcionalmente .DNG).

    Parsea y valida el contenedor TIFF, el MakerNote Nikon Type 3, el contexto del códec
    Lossless de 14 bits, el prefijo binario del NEF (1.7 MB) y la sonda colorimétrica
    una única vez. Ideal para inyectar decenas o cientos de imágenes sintéticas en
    evaluaciones de algoritmos de demosaicing sin repetir I/O sobre la plantilla.
    """

    def __init__(
        self,
        nef_path: str | Path,
        dng_path: str | Path | None = None,
        *,
        codec_backend: str = "auto",
    ) -> None:
        self.nef_path = Path(nef_path).resolve()
        if not self.nef_path.is_file():
            raise FileNotFoundError(f"No existe el NEF plantilla: {self.nef_path}")
        self.dng_path = Path(dng_path).resolve() if dng_path is not None else None
        if self.dng_path is not None and not self.dng_path.is_file():
            raise FileNotFoundError(f"No existe el DNG: {self.dng_path}")

        self.codec_backend = codec_backend

        # 1. Hash único del NEF y lectura única de estructura TIFF + MakerNote
        self.nef_sha256 = sha256_file(self.nef_path)
        self.tiff_reader = TiffReader(self.nef_path).parse()
        self.inspection_report = NefInspector(
            self.nef_path,
            compute_sha256=True,
            precomputed_sha256=self.nef_sha256,
            reader=self.tiff_reader,
        ).inspect()
        self.makernote_report = NikonMakerNoteInspector(
            self.nef_path,
            inspection=self.inspection_report,
            compute_sha256=True,
            precomputed_sha256=self.nef_sha256,
            reader=self.tiff_reader,
        ).inspect()
        self.lossless_ctx = NefLosslessContext.from_reports(
            self.nef_path,
            self.inspection_report,
            self.makernote_report,
            hash_check=True,
            precomputed_sha256=self.nef_sha256,
        )

        # 2. Cachear en RAM el prefijo del NEF anterior al strip RAW (~1.698 MB en D610)
        with self.nef_path.open("rb") as fd:
            self.cached_prefix_bytes = fd.read(self.lossless_ctx.raw_offset)

        self.patcher = NefContainerPatcher(
            self.lossless_ctx,
            self.inspection_report,
            self.makernote_report,
            cached_prefix_bytes=self.cached_prefix_bytes,
        )
        self.codec = NikonLosslessCodec(self.lossless_ctx.header, backend=codec_backend)
        self.bayer_target = BayerTarget.from_inspection(self.inspection_report)

        # 3. Sonda colorimétrica cacheada
        self.color_probe_report = NefCameraColorProbe().inspect(
            self.nef_path,
            compute_sha256=True,
            precomputed_sha256=self.nef_sha256,
            makernote_report=self.makernote_report,
        )

        # 4. Cachear hash y extractor del DNG si se proporcionó
        self.dng_sha256: str | None = None
        self._dng_extractor: DngAdobeProfileExtractor | None = None
        self._dng_cache_by_kelvin: dict[
            float | None, tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]
        ] = {}
        if self.dng_path is not None:
            self.dng_sha256 = sha256_file(self.dng_path)
            self._dng_extractor = DngAdobeProfileExtractor(self.dng_path)

    def get_dng_profile(
        self, kelvin: float | None = 6200.0
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
        if self._dng_extractor is None:
            raise ValueError("Esta NefSession se inicializó sin archivo DNG")
        key = float(kelvin) if kelvin is not None else None
        if key not in self._dng_cache_by_kelvin:
            self._dng_cache_by_kelvin[key] = self._dng_extractor.extract(
                kelvin=key, precomputed_sha256=self.dng_sha256
            )
        return self._dng_cache_by_kelvin[key]

    def inject(
        self,
        chart_path: str | Path | None = None,
        *,
        wb_gains: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        exposure_ev: float = -1.0,
        kelvin: float | None = 6200.0,
        black_dn: int = 0,
        white_dn: int = 15000,
        output_dir: str | Path = "experimentos_dng",
        run_dir: str | Path | None = None,
        verification_mode: str = "fast",
        persist_intermediates: bool = True,
        **kwargs: Any,
    ) -> PipelineResult:
        """Ejecuta una síntesis e inyección reutilizando los recursos cacheados de la sesión."""
        cfg = InjectionConfig(
            nef_path=self.nef_path,
            dng_path=self.dng_path,
            chart_path=Path(chart_path) if chart_path is not None else None,
            output_dir=Path(output_dir),
            run_dir=Path(run_dir) if run_dir is not None else None,
            wb_gains=wb_gains,
            exposure_ev=exposure_ev,
            kelvin=kelvin,
            black_dn=black_dn,
            white_dn=white_dn,
            verification_mode=verification_mode,
            codec_backend=self.codec_backend,
            persist_intermediates=persist_intermediates,
            **kwargs,
        )
        return inject_synthetic_nef(cfg, session=self)


def _run_verifications(
    candidate_nef: Path,
    payload_bytes: np.ndarray,
    reference_bayer: np.ndarray,
    session: NefSession,
    container_report: dict[str, Any],
    verification_mode: str,
) -> dict[str, Any]:
    """Ejecuta las comprobaciones post-inyección adaptándose al modo y librerías disponibles."""
    mode = verification_mode.lower()
    if mode not in ("fast", "paranoid", "full", "none"):
        raise InjectionError(f"verification_mode desconocido: {verification_mode}")

    verifications: dict[str, Any] = {}

    # 1. Codec roundtrip interno
    if mode in ("paranoid", "full"):
        rebuilt, used_bits = session.codec.decode(
            payload_bytes, session.lossless_ctx.width, session.lossless_ctx.height
        )
        bad = int(np.count_nonzero(rebuilt != reference_bayer))
        if bad:
            raise InjectionError(f"Codec roundtrip: {bad} fotositos discrepantes")
        verifications["codec_roundtrip"] = {
            "status": "pass",
            "sample_count": int(reference_bayer.size),
            "mismatched_samples": 0,
            "bits_consumed": int(used_bits),
            "stream_available_bits": 8 * int(payload_bytes.size),
            "padding_bits": 8 * int(payload_bytes.size) - int(used_bits),
        }
    elif mode == "fast":
        # En modo fast: si Numba está disponible, el roundtrip completo tarda solo ~0.16 s;
        # si no está Numba pero tampoco rawpy, verificamos las primeras 64 filas (~0.05 s).
        if numba_acceleration_available():
            rebuilt, used_bits = session.codec.decode(
                payload_bytes, session.lossless_ctx.width, session.lossless_ctx.height
            )
            bad = int(np.count_nonzero(rebuilt != reference_bayer))
            if bad:
                raise InjectionError(f"Codec roundtrip: {bad} fotositos discrepantes")
            verifications["codec_roundtrip"] = {
                "status": "pass",
                "sample_count": int(reference_bayer.size),
                "mismatched_samples": 0,
                "bits_consumed": int(used_bits),
                "stream_available_bits": 8 * int(payload_bytes.size),
                "padding_bits": 8 * int(payload_bytes.size) - int(used_bits),
            }
        else:
            sample_rows = min(64, session.lossless_ctx.height)
            rebuilt_sub, used_bits_sub = session.codec.decode(
                payload_bytes, session.lossless_ctx.width, sample_rows
            )
            bad = int(np.count_nonzero(rebuilt_sub != reference_bayer[:sample_rows]))
            if bad:
                raise InjectionError(f"Codec prefix roundtrip ({sample_rows} rows): {bad} discrepancias")
            verifications["codec_roundtrip"] = {
                "status": "pass_prefix_fast",
                "verified_rows": sample_rows,
                "sample_count": int(rebuilt_sub.size),
                "mismatched_samples": 0,
                "bits_consumed_prefix": int(used_bits_sub),
            }
    else:
        verifications["codec_roundtrip"] = {"status": "skipped", "reason": "verification_mode=none"}

    # 2. Container check (ya calculado durante la escritura en memoria)
    verifications["container"] = container_report

    # 3. LibRaw & rawpy checks (en una sola apertura de rawpy cuando es posible)
    if mode == "none":
        verifications["libraw_unpack"] = {"status": "skipped"}
        verifications["rawpy_pixels"] = {"status": "not_performed"}
        return verifications

    # Intentar verificación píxel a píxel con rawpy (valida simultáneamente LibRaw unpack y las 24.49M muestras)
    rawpy_result: dict[str, Any] | None = None
    try:
        import rawpy

        with rawpy.imread(str(candidate_nef)) as raw:
            actual = raw.raw_image
            if actual.shape != reference_bayer.shape:
                raise InjectionError(
                    f"Dimensiones decodificadas por rawpy {actual.shape}, esperadas {reference_bayer.shape}"
                )
            count = int(np.count_nonzero(actual != reference_bayer))
            if count:
                raise InjectionError(f"Comparación externa rawpy: {count} fotositos discrepantes")
            rawpy_result = {
                "status": "pass",
                "sample_count": int(reference_bayer.size),
                "mismatched_samples": 0,
                "rawpy_version": str(rawpy.__version__),
                "libraw_version": str(rawpy.libraw_version),
            }
    except ImportError:
        rawpy_result = None

    if mode in ("paranoid", "full"):
        try:
            probe = LibRawNativeProbe().probe(candidate_nef)
            if not probe.get("unpack_success") or (
                probe.get("raw_width"),
                probe.get("raw_height"),
            ) != (session.lossless_ctx.width, session.lossless_ctx.height):
                raise InjectionError("LibRaw nativo devolvió dimensiones o estado inválidos")
            verifications["libraw_unpack"] = {
                "status": "pass",
                "details": probe,
                "note": "Unpack exitoso; NO compara los valores de los fotositos.",
            }
        except PayloadDecodeError:
            if rawpy_result is not None:
                verifications["libraw_unpack"] = {
                    "status": "pass",
                    "api": "rawpy_embedded_libraw_fallback",
                    "rawpy_exact_pixels": rawpy_result,
                    "note": "LibRaw C no encontrado; rawpy validó todas las muestras.",
                }
            else:
                verifications["libraw_unpack"] = {
                    "status": "skipped_not_installed",
                    "note": "Ni libraw nativo ni rawpy están instalados en este entorno.",
                }
    else:
        # En modo fast, evitamos abrir y descomprimir el NEF dos veces si rawpy ya validó los 24.49M píxeles
        if rawpy_result is not None:
            verifications["libraw_unpack"] = {
                "status": "pass",
                "api": "verified_via_rawpy_single_pass",
                "raw_width": session.lossless_ctx.width,
                "raw_height": session.lossless_ctx.height,
                "unpack_success": True,
                "note": "Validado en la misma pasada de rawpy.imread sin doble descompresión.",
            }
        else:
            try:
                probe = LibRawNativeProbe().probe(candidate_nef)
                verifications["libraw_unpack"] = {
                    "status": "pass",
                    "details": probe,
                    "note": "Unpack exitoso; NO compara los valores de los fotositos.",
                }
            except PayloadDecodeError:
                verifications["libraw_unpack"] = {
                    "status": "skipped_not_installed",
                    "note": "Ni libraw nativo ni rawpy están instalados en este entorno.",
                }

    if rawpy_result is not None:
        verifications["rawpy_pixels"] = rawpy_result
    else:
        verifications["rawpy_pixels"] = {
            "status": "skipped_not_installed",
            "note": "rawpy no instalado; se validó contenedor y códec interno.",
        }

    return verifications


def render_validation_srgb8_png(nef_path: str | Path, output_png: str | Path) -> Path | None:
    """Renderiza una vista previa sRGB de 8 bits usando rawpy.postprocess si rawpy está instalado."""
    try:
        import rawpy
    except ImportError:
        return None
    out = Path(output_png)
    with rawpy.imread(str(nef_path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    Image.fromarray(rgb, mode="RGB").save(out, format="PNG")
    return out


def inject_synthetic_nef(
    config: InjectionConfig | None = None,
    *,
    session: NefSession | None = None,
    **kwargs: Any,
) -> PipelineResult:
    """Ejecuta el pipeline completo de síntesis Bayer e inyección transaccional en un NEF Nikon.

    Puede invocarse pasando un `InjectionConfig` o directamente argumentos con nombre
    (`nef_path=...`, `dng_path=...`, `chart_path=...`, etc.).
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    if config is None:
        config = InjectionConfig(**kwargs)
    elif kwargs:
        raise ValueError("No mezclar objeto config con kwargs adicionales")

    # 1. Inicializar o reutilizar sesión NEF/DNG
    t0 = time.perf_counter()
    if session is None:
        session = NefSession(
            config.nef_path,
            config.dng_path,
            codec_backend=config.codec_backend,
        )
    timings["01_session_and_inspection_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 2. Preparar directorio de ejecución
    stem = session.nef_path.stem
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    run_id = f"{now_str}_{uuid.uuid4().hex[:8]}"
    if config.run_dir is not None:
        run_dir = Path(config.run_dir).resolve()
    else:
        out_root = Path(config.output_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        tag_type = "AdobeDNG" if session.dng_path is not None else "LibRawProbe"
        run_dir = out_root / f"{stem}__{tag_type}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Resolver carta de color (o autogenerar el patrón seleccionado)
    t0 = time.perf_counter()
    if config.chart_path is not None:
        chart_path = Path(config.chart_path).resolve()
        if not chart_path.is_file():
            raise FileNotFoundError(f"No existe la carta de entrada: {chart_path}")
    elif config.pattern == "star":
        chart_path = generate_siemens_star_png(run_dir / "siemens_star_srgb.png")
    else:
        chart_path = generate_colorchecker24_png(run_dir / "colorchecker24_srgb.png")
    chart_sha256 = sha256_file(chart_path)
    timings["02_chart_prepare_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 4. Extraer perfil DNG (o usar matriz candidata de LibRaw si no se suministró DNG)
    t0 = time.perf_counter()
    dng_report: dict[str, Any] | None = None
    dng_luts: dict[str, np.ndarray] | None = None
    if session.dng_path is not None:
        dng_report, rgb_to_cam, dng_luts = session.get_dng_profile(config.kelvin)
        verify_nef_dng_provenance(
            session.inspection_report,
            dng_report,
            session.nef_path,
            expected_profile=config.expected_profile,
            allow_different_original_name=config.allow_different_original_name,
        )
    else:
        rgb_to_cam = np.asarray(
            session.color_probe_report["color"]["candidate_linear_srgb_to_camera_3x3"],
            dtype=np.float64,
        )
    timings["03_color_profile_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 4b. Simulación óptica pre-CFA en RAM (opcional: PSF, OLPF, viñeteado, CA lateral, distorsión)
    optics_manifest: dict[str, Any] | None = None
    optics_origin: str | None = None
    optics_rgb_linear: np.ndarray | None = None
    optics_tiff_path = run_dir / "optica_pre_cfa_linear.tif"
    optics_json_path = run_dir / "optica_pre_cfa_linear.json"
    if config.optics_profile is not None:
        t0 = time.perf_counter()
        optics_cfg, optics_origin = OpticsConfig.load(config.optics_profile)
        if config.save_optics_tiff:
            optics_manifest = simulate_optics(
                chart_path,
                optics_tiff_path,
                optics_cfg,
                width=session.bayer_target.width,
                height=session.bayer_target.height,
                resize=config.resize_mode,
                orientation=config.orientation,
                rotation=config.rotation,
                encoding=config.input_encoding,
            )
        else:
            optics_rgb_linear, optics_manifest = simulate_optics_in_memory(
                chart_path,
                optics_cfg,
                width=session.bayer_target.width,
                height=session.bayer_target.height,
                resize=config.resize_mode,
                orientation=config.orientation,
                rotation=config.rotation,
                encoding=config.input_encoding,
                precomputed_source_sha256=chart_sha256,
            )
            if config.persist_intermediates:
                write_json_atomic(optics_json_path, optics_manifest, overwrite=True)
        timings["03b_optics_simulation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 5. Síntesis del mosaico Bayer Ideal 2D uint16 en memoria
    t0 = time.perf_counter()
    if optics_manifest is not None:
        bayer_cfg = BayerConversionConfig(
            resize="none",
            input_encoding="linear",
            black_dn=config.black_dn,
            white_dn=config.white_dn,
            background_linear=config.background_linear,
            exposure_ev=config.exposure_ev,
            gains=config.wb_gains,
            rgb_to_camera=tuple(tuple(float(v) for v in row) for row in rgb_to_cam),
        )
        converter = ImageToBayerConverter(session.bayer_target, bayer_cfg)
        if optics_rgb_linear is not None:
            bayer_array, bayer_manifest = converter.synthesize_from_linear_rgb(
                optics_rgb_linear,
                source_filename=optics_manifest["output"]["filename"],
                source_sha256=optics_manifest["output"]["sha256"],
                optical_psf_applied=True,
            )
            del optics_rgb_linear
        else:
            bayer_array, bayer_manifest = converter.synthesize_in_memory(
                optics_tiff_path,
                precomputed_source_sha256=optics_manifest["output"]["sha256"],
            )
        bayer_manifest["processing"]["orientation"] = optics_manifest["processing"]["orientation"]
    else:
        bayer_cfg = BayerConversionConfig(
            resize=config.resize_mode,
            orientation=config.orientation,
            rotation=config.rotation,
            input_encoding=config.input_encoding,
            black_dn=config.black_dn,
            white_dn=config.white_dn,
            background_linear=config.background_linear,
            exposure_ev=config.exposure_ev,
            gains=config.wb_gains,
            rgb_to_camera=tuple(tuple(float(v) for v in row) for row in rgb_to_cam),
        )
        converter = ImageToBayerConverter(session.bayer_target, bayer_cfg)
        bayer_array, bayer_manifest = converter.synthesize_in_memory(
            chart_path, precomputed_source_sha256=chart_sha256
        )

    if config.persist_intermediates:
        bayer_npy_path = run_dir / "bayer_final.npy"
        bayer_sha256 = save_npy_and_sha256(bayer_npy_path, bayer_array, overwrite=True)
        bayer_manifest["array"]["sha256"] = bayer_sha256
        if config.save_npz:
            npz_path = run_dir / "bayer_final.npz"
            np.savez_compressed(npz_path, raw=bayer_array)
            bayer_manifest["archive"] = {
                "filename": npz_path.name,
                "key": "raw",
                "sha256": sha256_file(npz_path),
            }
        write_json_atomic(run_dir / "bayer_final.json", bayer_manifest, overwrite=True)
    else:
        bayer_sha256 = sha256_npy_in_memory(bayer_array)
        bayer_manifest["array"]["sha256"] = bayer_sha256
    timings["04_bayer_synthesis_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 5b. Simulación de artefactos optoelectrónicos de sensor post-CFA en RAM (opcional)
    sensor_manifest: dict[str, Any] | None = None
    sensor_origin: str | None = None
    bayer_for_injection = bayer_array
    injected_bayer_sha256 = bayer_sha256
    injected_bayer_name = "bayer_final.npy"
    if config.sensor_profile is not None:
        t0 = time.perf_counter()
        sensor_cfg, sensor_origin = SensorConfig.load(config.sensor_profile)
        bayer_for_injection, sensor_manifest = NefSensorSimulator(sensor_cfg).simulate_in_memory(
            bayer_array,
            source_meta=bayer_manifest,
            source_sha256=bayer_sha256,
            source_filename="bayer_final.npy",
            source_manifest_path=(
                str(run_dir / "bayer_final.json") if config.persist_intermediates else None
            ),
            output_filename="bayer_sensor.npy",
        )
        injected_bayer_sha256 = sensor_manifest["array"]["sha256"]
        injected_bayer_name = "bayer_sensor.npy"
        if config.persist_intermediates:
            save_npy_and_sha256(run_dir / "bayer_sensor.npy", bayer_for_injection, overwrite=True)
            write_json_atomic(run_dir / "bayer_sensor.json", sensor_manifest, overwrite=True)
        timings["04b_sensor_simulation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 6. Codificación Nikon 14-bit Lossless en memoria
    t0 = time.perf_counter()
    payload_bytes, bits_used, engine_used = session.codec.encode_in_memory(bayer_for_injection)
    payload_sha256 = sha256_bytes(payload_bytes)
    timings["05_lossless_encode_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 6b. Regenerar los previews desde --graft tras aplicar la geometría visible del RAW.
    # No se derivan del mosaico Bayer ni del perfil de sensor: no existe una conversión
    # de revelado Nikon inequívoca para ello y el usuario suministró explícitamente graft.
    t0 = time.perf_counter()
    preview_rgb8, preview_manifest = prepare_preview_rgb8(
        chart_path,
        target_width=session.bayer_target.width,
        target_height=session.bayer_target.height,
        resize=config.resize_mode,
        orientation=config.orientation,
        rotation=config.rotation,
        input_encoding=config.input_encoding,
        background_linear=config.background_linear,
        exposure_ev=config.exposure_ev,
        precomputed_source_sha256=chart_sha256,
    )
    preview_payloads = session.patcher.build_preview_payloads(preview_rgb8)
    preview_manifest["embedded_outputs"] = [payload.as_report() for payload in preview_payloads]
    del preview_rgb8
    timings["05b_embedded_preview_render_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 7. Construcción del contenedor NEF + Validación
    t0 = time.perf_counter()
    plan = session.patcher.plan(
        int(payload_bytes.size),
        strategy=config.strategy,
        preview_payloads=preview_payloads,
    )
    out_nef_name = (
        f"{stem}_AdobeStandard_sintetico.NEF"
        if session.dng_path is not None
        else f"{stem}_sintetico.NEF"
    )
    output_nef_path = run_dir / out_nef_name
    injection_json_path = run_dir / f"{Path(out_nef_name).stem}.injection.json"

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_nef_path.stem}.",
        suffix=".partial",
        dir=run_dir,
        delete=False,
    ) as tmp_f:
        candidate = Path(tmp_f.name)

    try:
        output_nef_sha256, container_report = session.patcher.build_and_validate_in_memory(
            payload_bytes,
            payload_sha256,
            candidate,
            plan,
            preview_payloads=preview_payloads,
        )
        timings["06_container_build_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        t0 = time.perf_counter()
        verifications = _run_verifications(
            candidate,
            payload_bytes,
            bayer_for_injection,
            session,
            container_report,
            config.verification_mode,
        )
        timings["07_verification_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        injection_report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tool": {"name": "nef_injector", "version": "0.5.0"},
            "source": {
                "filename": session.nef_path.name,
                "size_bytes": session.nef_path.stat().st_size,
                "sha256": session.nef_sha256,
            },
            "input": {
                "raw_npy": injected_bayer_name,
                "raw_npy_sha256": injected_bayer_sha256,
                "shape_hw": [session.lossless_ctx.height, session.lossless_ctx.width],
                "dtype": "uint16",
                "payload_origin": "encoded_from_npy",
                "numba_acceleration_used": engine_used == "numba",
                "codec_engine": engine_used,
                "compressed_payload_sha256": payload_sha256,
            },
            "output": {
                "filename": output_nef_path.name,
                "size_bytes": plan.output_size,
                "sha256": output_nef_sha256,
                "nef_compatibility": "not_tested_in_commercial_raw_editors",
            },
            "embedded_previews": preview_manifest,
            "plan": plan.as_report(),
            "verification": verifications,
            "known_limitations": [
                "Sólo Nikon D610 14-bit Lossless encabezamiento 0x4630.",
                "Los previews usan una conversión sRGB de --graft; no son un revelado Nikon del Bayer inyectado.",
                "La compatibilidad con Adobe Camera Raw y Capture One debe comprobarse externamente.",
                "Un unpack exitoso de LibRaw no equivale a igualdad de píxeles: ejecutar --verify-rawpy.",
            ],
        }

        if output_nef_path.exists():
            output_nef_path.unlink()
        os.link(candidate, output_nef_path)
    finally:
        candidate.unlink(missing_ok=True)

    # 8. Persistir artefactos de trazabilidad científica
    t0 = time.perf_counter()
    inspection_path = run_dir / "original.inspection.json"
    maker_path = run_dir / "original.makernote.json"
    probe_path = run_dir / "original.color_probe.json"
    probe_matrix_path = run_dir / "original.libraw_matrix_DIAGNOSTICO.json"
    dng_report_path = run_dir / "perfil_dng.dng_profile_report.json"
    dng_matrix_path = run_dir / "perfil_dng.adobe_standard_rgb_to_camera.json"
    dng_luts_path = run_dir / "perfil_dng.dng_profile_luts.npz"
    layout_path = run_dir / "colorchecker.layout.json"
    summary_path = run_dir / "resumen_ejecucion.json"

    write_json_atomic(injection_json_path, injection_report, overwrite=True)

    if config.persist_intermediates:
        write_json_atomic(inspection_path, session.inspection_report, overwrite=True)
        write_json_atomic(maker_path, session.makernote_report, overwrite=True)
        write_json_atomic(probe_path, session.color_probe_report, overwrite=True)
        write_json_atomic(
            probe_matrix_path,
            session.color_probe_report["color"]["candidate_linear_srgb_to_camera_3x3"],
            overwrite=True,
        )
        if dng_report is not None and dng_luts is not None:
            dng_matrix_sha256 = write_json_atomic(dng_matrix_path, rgb_to_cam.tolist(), overwrite=True)
            write_json_atomic(dng_report_path, dng_report, overwrite=True)
            np.savez_compressed(dng_luts_path, **dng_luts)
        else:
            dng_matrix_sha256 = write_json_atomic(dng_matrix_path, rgb_to_cam.tolist(), overwrite=True)

        if config.custom_layout_path is not None:
            shutil.copyfile(config.custom_layout_path, layout_path)
        else:
            layout_obj = make_grid_layout(
                config.layout_cols,
                config.layout_rows,
                config.layout_roi_fill,
                config.layout_neutrals_last_row,
            )
            write_json_atomic(layout_path, layout_obj, overwrite=True)

        if config.decode_original:
            raw_orig_dir = run_dir / "raw_original"
            raw_orig_dir.mkdir(parents=True, exist_ok=True)
            frame = RawPyBackend().load(session.nef_path)
            save_npy_and_sha256(
                raw_orig_dir / f"{stem}.raw_full.npy", frame.image, overwrite=True
            )
            save_npy_and_sha256(
                raw_orig_dir / f"{stem}.raw_visible.npy", frame.visible, overwrite=True
            )

        if config.render_validation_png:
            render_validation_srgb8_png(output_nef_path, run_dir / "validacion_rawpy_srgb8.png")
    else:
        dng_matrix_sha256 = sha256_bytes(
            (str(rgb_to_cam.tolist())).encode("utf-8")
        )

    etapas = [
        "01. Inspeccion NEF original",
        "02. Nikon MakerNote original",
        "04. Probe LibRaw de diagnostico (NO usado en sintesis)",
        "05. Extraer matrices y LUT del DNG Adobe",
    ]
    if optics_manifest is not None:
        etapas.append("06b. Simular PSF, OLPF, vignette y CA PRE-CFA")
    etapas.append("07. Crear Bayer con matriz DNG y WB/EV")
    if sensor_manifest is not None:
        etapas.append("08b. Simular ruido y no uniformidad del sensor")
    etapas.extend(
        [
            "08c. Regenerar previews TIFF/JPEG desde graft",
            "09. Inyectar y verificar NEF sintetico",
            "10. Crear layout de parches",
        ]
    )
    xmp_profile = (
        dng_report.get("xmp_default_camera_profile", {}).get("xmp_camera_profile_field")
        if dng_report
        else None
    )
    summary_report: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "estado": "nef_generado",
        "fecha": dt.datetime.now().astimezone().isoformat(),
        "run_id": run_id,
        "etapas": etapas,
        "entradas": {
            "nef": str(session.nef_path),
            "nef_sha256": session.nef_sha256,
            "dng": str(session.dng_path) if session.dng_path else None,
            "dng_sha256": session.dng_sha256,
            "carta": str(chart_path),
            "carta_sha256": chart_sha256,
            "patron_sintetico": config.pattern if config.chart_path is None else None,
            "nombre_original_dng": (
                dng_report["camera"]["original_nef_filename"] if dng_report else None
            ),
            "modelo_nef": session.inspection_report["camera"]["model"],
            "modelo_dng": dng_report["camera"]["model"] if dng_report else None,
        },
        "parametros": {
            "perfil_embebido": (
                dng_report["embedded_profile"]["profile_name"] if dng_report else "LibRaw_Candidate"
            ),
            "perfil_xmp": xmp_profile,
            "fuente_matriz": (
                "DNG ForwardMatrix 1/2, AsShotNeutral, CameraCalibration, AnalogBalance, sRGB D50"
                if dng_report
                else "LibRaw inverse(rgb_cam) + AsShot WB"
            ),
            "temperatura_interpolacion_kelvin": (
                dng_report["derived_linear_model"]["interpolation_temperature_K"]
                if dng_report
                else config.kelvin
            ),
            "wb_r_g1_g2_b": list(config.wb_gains),
            "exposicion_ev": config.exposure_ev,
            "black_dn": config.black_dn,
            "white_dn": config.white_dn,
            "resize": config.resize_mode,
            "input_encoding": config.input_encoding,
            "estrategia": config.strategy,
            "rawpy_verificacion": verifications.get("rawpy_pixels", {}).get("status") == "pass",
            "verification_mode": config.verification_mode,
            "codec_engine": engine_used,
            "sensor_simulation_enabled": sensor_manifest is not None,
            "optics_simulation_enabled": optics_manifest is not None,
        },
        "artefactos": {
            "carpeta": str(run_dir),
            "perfil_dng": str(dng_report_path) if dng_report else None,
            "matriz_inversa_dng": str(dng_matrix_path),
            "matriz_sha256": dng_matrix_sha256,
            "luts_dng": str(dng_luts_path) if dng_report else None,
            "inspeccion": str(inspection_path),
            "makernote": str(maker_path),
            "probe_libraw_diagnostico": str(probe_path),
            "bayer": str(run_dir / "bayer_final.npy"),
            "bayer_sha256": bayer_sha256,
            "manifiesto": str(run_dir / "bayer_final.json"),
            "optics_profile": optics_origin,
            "optics_tiff_linear": (
                str(optics_tiff_path)
                if (optics_manifest is not None and config.save_optics_tiff)
                else None
            ),
            "optics_manifest": (
                str(optics_json_path)
                if (optics_manifest is not None and config.persist_intermediates)
                else None
            ),
            "sensor_profile": sensor_origin,
            "bayer_inyectado": str(run_dir / injected_bayer_name),
            "bayer_inyectado_sha256": injected_bayer_sha256,
            "bayer_sensor_manifest": (
                str(run_dir / "bayer_sensor.json")
                if (sensor_manifest is not None and config.persist_intermediates)
                else None
            ),
            "nef_final": str(output_nef_path),
            "informe_inyeccion": str(injection_json_path),
            "layout": str(layout_path),
        },
        "limitaciones": [
            "La matriz es la inversa aproximada de la ETAPA MATRICIAL del perfil DNG, NO de todo Adobe Camera Raw.",
            "Las LUT y transformaciones tonales de Adobe no se invierten en esta version.",
            "El DNG puede tener distinto recorte, escala y negro que el NEF. BlackDN/WhiteDN son elegidos experimentalmente.",
            "La coincidencia del nombre original no garantiza la identidad RAW del NEF y del DNG.",
            "Las miniaturas y previews se generan desde --graft en sRGB; no reproducen el revelado Nikon exacto del Bayer inyectado.",
            "La matriz DCRAW de Nikon D610 coincide con ColorMatrix2 del DNG, que es XYZ->camara, no directamente sRGB->RAW.",
            "Los modelos de ruido y optica ilustrativos requieren caracterizacion experimental por ISO, iluminante y objetivo.",
            "El perfil XMP puede ser distinto del Adobe Standard incrustado; fijar Adobe Standard al comparar TIFF.",
            "Una matriz 3x3 sola no corrige dominantes distintas entre diferentes niveles de gris.",
        ],
    }

    if config.compare_with_dir is not None and config.persist_intermediates:
        prior_dir = Path(config.compare_with_dir).resolve()
        prior_npy = prior_dir / "bayer_final.npy"
        prior_nef = prior_dir / out_nef_name
        summary_report["comparacion_experimento_anterior"] = compare_bayer_syntheses(
            prior_npy,
            run_dir / "bayer_final.npy",
            nef_base=prior_nef if prior_nef.is_file() else None,
            nef_test=output_nef_path,
        )

    timings["08_persist_reports_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
    summary_report["rendimiento_ms"] = timings

    write_json_atomic(summary_path, summary_report, overwrite=True)

    return PipelineResult(
        run_id=run_id,
        run_dir=run_dir,
        output_nef_path=output_nef_path,
        output_nef_sha256=output_nef_sha256,
        bayer_sha256=bayer_sha256,
        compressed_payload_sha256=payload_sha256,
        compressed_payload_bytes=int(payload_bytes.size),
        rgb_to_camera_matrix=rgb_to_cam,
        summary_report=summary_report,
        injection_report=injection_report,
        bayer_manifest=bayer_manifest,
        dng_profile_report=dng_report,
        color_probe_report=session.color_probe_report,
        inspection_report=session.inspection_report,
        makernote_report=session.makernote_report,
        timings_ms=timings,
        injected_bayer_sha256=injected_bayer_sha256,
        optics_manifest=optics_manifest,
        sensor_manifest=sensor_manifest,
    )


def analyze_rendered_tiff(
    run_dir: str | Path,
    rendered_tiff: str | Path,
    *,
    custom_layout_path: str | Path | None = None,
    encoding: str = "srgb",
    ignore_icc: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Ejecuta el análisis cuantitativo ΔE00 de un TIFF revelado contra un experimento generado."""
    folder = Path(run_dir).resolve()
    summary_path = folder / "resumen_ejecucion.json"
    meta = read_json(summary_path)
    if meta.get("estado") != "nef_generado":
        raise RuntimeError(f"Ejecución no completada: {meta.get('estado')}")
    reference = Path(meta["entradas"]["carta"]).resolve()
    if sha256_file(reference) != meta["entradas"]["carta_sha256"]:
        raise RuntimeError("La carta RGB original ha cambiado; comparación colorimétrica no trazable.")
    tiff_path = Path(rendered_tiff).resolve()
    layout_path = (
        Path(custom_layout_path).resolve()
        if custom_layout_path is not None
        else folder / "colorchecker.layout.json"
    )
    sensor_bayer = folder / "bayer_sensor.npy"
    bayer_path = sensor_bayer if sensor_bayer.is_file() else (folder / "bayer_final.npy")
    manifest_path = (
        folder / "bayer_sensor.json"
        if sensor_bayer.is_file()
        else (folder / "bayer_final.json")
    )
    probe_path = folder / "original.color_probe.json"

    profile = CameraColorProfile.from_file(probe_path) if probe_path.is_file() else None
    manifest = read_json(manifest_path)
    layout = read_json(layout_path)

    report = NefColorCalibrator(profile).analyze(
        reference,
        tiff_path,
        layout,
        raw_path=bayer_path,
        manifest=manifest,
        encoding=encoding,
        ignore_icc=ignore_icc,
    )
    report["source"]["generator_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    now_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3] + "_" + uuid.uuid4().hex[:6]
    report_path = folder / f"calibracion_TIFF_{now_tag}.json"
    write_json_atomic(report_path, report, overwrite=True)
    return report_path, report

