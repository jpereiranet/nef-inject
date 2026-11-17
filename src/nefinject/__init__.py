"""nefinject — Librería científica de alto rendimiento para síntesis CFA Bayer
e inyección transaccional en contenedores RAW Nikon NEF.

`nefinject` es un framework modular en Python diseñado para la experimentación,
simulación física y procesamiento científico de imágenes en formato RAW de cámaras
digitales (especializado en contenedores Nikon NEF TIFF-EP de 14 bits Lossless).

Capacidades principales de la librería:
- Contenedores TIFF/EP y MakerNote: Inspección de bajo nivel de IFDs, metadatos EXIF
  y MakerNote Nikon Type 3, junto con inyección binaria transaccional con verificación
  estricta de sumas de control criptográficas (SHA-256).
- Códec Nikon Lossless (0x4630): Compresión y descompresión canónica de Huffman
  para diferencias predictivas DPCM por paridad de canal, acelerada mediante NumPy
  vectorizado y compilación JIT (Numba).
- Colorimetría y Perfiles Adobe DNG: Extracción e interpolación de matrices de color,
  matrices de avance (ForwardMatrix) y adaptación cromática de Bradford bajo iluminantes
  estándar A y D65, con calibración empírica de temperatura correlacionada (CCT).
- Síntesis Bayer CFA: Conversión de imágenes de entrada a mosaicos Bayer RGGB/BGGR de
  14 bits con corrección gamma sRGB EOTF (IEC 61966-2-1), ajuste de exposición (EV)
  y generación de patrones sintéticos de prueba (ColorChecker, barridos de frecuencia).
- Simulación Óptica Pre-CFA: Modelado físico de difracción y desenfoque (PSF), filtro
  antialiasing paso bajo óptico (OLPF), aberración cromática transversal (lateral CA),
  viñeteado óptico cos^4(theta) y distorsión geométrica radial.
- Simulación de Sensor Post-CFA: Modelado optoelectrónico de ruido de patrón fijo (DSNU,
  PRNU), píxeles defectuosos (calientes/muertos), ruido térmico de lectura y ruido de
  disparo fotónico (Poisson), junto con caracterización empírica por darks/flats.
- Calibración y Métrica CIEDE2000: Cuantificación colorimétrica rigurosa de errores Delta E
  (CIE 2000) sobre cartas de color Macbeth ColorChecker de 24 parches.
- Orquestación End-to-End: Pipeline completo de ejecución en memoria de alto rendimiento
  sin archivos intermedios redundantes en disco.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

from .calibration import (
    BayerRawSampler,
    CalibrationRefiner,
    CameraColorProfile,
    ChartMeasurements,
    ColorImage,
    ColorTransformCalibrator,
    ExposureCalibrator,
    NefColorCalibrator,
    WhiteBalanceCalibrator,
    compare_bayer_syntheses,
    delta_e_2000,
    linear_srgb_to_lab,
    linear_to_srgb,
    srgb_to_linear,
)
from .codec import (
    CanonicalHuffman,
    NikonLosslessCodec,
    encode_nikon_lossless_numpy,
    numba_acceleration_available,
)
from .color import (
    DCRAW_D610_COLOR_MATRIX_2,
    SRGB_TO_XYZ_D50,
    DngAdobeProfileExtractor,
    NefCameraColorProbe,
    approx_cct_mccamy,
    estimate_temperature_dng,
    mix_weight_dng,
    verify_nef_dng_provenance,
)
from .container import (
    CfaGeometry,
    InjectionPlan,
    LibRawNativeProbe,
    LosslessHeader,
    MakerNoteIFDReader,
    MakerNoteTag,
    NefContainerPatcher,
    NefInspector,
    NefLosslessContext,
    NikonMakerNoteInspector,
    PatchField,
    RawFrame,
    RawPyBackend,
    TiffDirectory,
    TiffReader,
    TiffTag,
)
from .models import (
    CalibrationError,
    CameraColorProbeError,
    CodecError,
    ConversionError,
    DngProfileError,
    InjectionConfig,
    InjectionError,
    LIBRARY_VERSION,
    MakerNoteParseError,
    NefInjectError,
    PayloadDecodeError,
    PipelineResult,
    TiffParseError,
    read_json,
    save_npy_and_sha256,
    sha256_bytes,
    sha256_file,
    sha256_npy_in_memory,
    write_json_atomic,
)
from .optics import (
    Distortion,
    LateralCA,
    OLPF,
    OpticsConfig,
    OpticsError,
    PSF,
    Vignetting,
    apply_optics,
    simulate_optics,
    simulate_optics_in_memory,
)
from .pipeline import (
    NefSession,
    analyze_rendered_tiff,
    inject_synthetic_nef,
    render_validation_srgb8_png,
)
from .profiles import PROFILES_DIR
from .sensor import (
    Defects,
    FixedMaps,
    NefSensorSimulator,
    SensorConfig,
    SensorSimulationError,
    Vignette,
    characterize,
    characterize_sensor,
)
from .synthesis import (
    BayerConversionConfig,
    BayerPatternConfig,
    BayerPatternGenerator,
    BayerTarget,
    ImageToBayerConverter,
    generate_colorchecker24_png,
    generate_siemens_star_png,
    load_image,
    make_grid_layout,
)

__version__ = LIBRARY_VERSION

__all__ = [
    "__version__",
    # Orquestación principal y sesiones
    "NefSession",
    "inject_synthetic_nef",
    "analyze_rendered_tiff",
    "render_validation_srgb8_png",
    "InjectionConfig",
    "PipelineResult",
    # Contenedor TIFF / NEF / MakerNote
    "TiffReader",
    "TiffDirectory",
    "TiffTag",
    "NefInspector",
    "MakerNoteIFDReader",
    "MakerNoteTag",
    "NikonMakerNoteInspector",
    "LosslessHeader",
    "NefLosslessContext",
    "PatchField",
    "InjectionPlan",
    "NefContainerPatcher",
    "LibRawNativeProbe",
    "RawPyBackend",
    "RawFrame",
    "CfaGeometry",
    # Colorimetría y Perfiles DNG
    "NefCameraColorProbe",
    "DngAdobeProfileExtractor",
    "verify_nef_dng_provenance",
    "SRGB_TO_XYZ_D50",
    "DCRAW_D610_COLOR_MATRIX_2",
    "approx_cct_mccamy",
    "mix_weight_dng",
    "estimate_temperature_dng",
    # Síntesis Bayer y Cartas de Color
    "BayerTarget",
    "BayerConversionConfig",
    "ImageToBayerConverter",
    "BayerPatternConfig",
    "BayerPatternGenerator",
    "generate_colorchecker24_png",
    "generate_siemens_star_png",
    "make_grid_layout",
    "load_image",
    # Códec Nikon 14-bit Lossless
    "CanonicalHuffman",
    "NikonLosslessCodec",
    "encode_nikon_lossless_numpy",
    "numba_acceleration_available",
    # Calibración y Métricas Colorimétricas
    "NefColorCalibrator",
    "CalibrationRefiner",
    "CameraColorProfile",
    "ChartMeasurements",
    "WhiteBalanceCalibrator",
    "ExposureCalibrator",
    "ColorTransformCalibrator",
    "BayerRawSampler",
    "ColorImage",
    "compare_bayer_syntheses",
    "delta_e_2000",
    "linear_srgb_to_lab",
    "srgb_to_linear",
    "linear_to_srgb",
    # Simulación Óptica Pre-CFA y de Sensor Post-CFA
    "OpticsConfig",
    "PSF",
    "OLPF",
    "Vignetting",
    "LateralCA",
    "Distortion",
    "apply_optics",
    "simulate_optics",
    "simulate_optics_in_memory",
    "OpticsError",
    "SensorConfig",
    "Vignette",
    "Defects",
    "FixedMaps",
    "NefSensorSimulator",
    "characterize_sensor",
    "characterize",
    "SensorSimulationError",
    "PROFILES_DIR",
    # Utilidades y Excepciones
    "sha256_file",
    "sha256_bytes",
    "sha256_npy_in_memory",
    "save_npy_and_sha256",
    "read_json",
    "write_json_atomic",
    "NefInjectError",
    "TiffParseError",
    "MakerNoteParseError",
    "PayloadDecodeError",
    "CameraColorProbeError",
    "DngProfileError",
    "ConversionError",
    "CodecError",
    "InjectionError",
    "CalibrationError",
]

