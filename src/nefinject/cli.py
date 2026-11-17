"""Interfaz de línea de comandos (CLI) de la librería nefinject.

Proporciona comandos de consola para ejecutar el pipeline integral de síntesis
e inyección RAW, así como operaciones atómicas de simulación, caracterización
e inspección forense de archivos Nikon NEF.

Comandos principales disponibles:
- `nefinject run`: Ejecuta el flujo completo de extremo a extremo:
    nefinject run --nef plantilla.NEF --dng perfil.DNG --graft carta.png \\
        --wb 1.0,1.0,1.0,1.0 --ev -1.0 --kelvin 6200 --output-dir resultados/
- `nefinject analyze-tiff`: Analiza un revelado TIFF contra la carta de referencia y calcula métricas CIEDE2000:
    nefinject analyze-tiff --run-dir resultados/experimento_01/ --tiff revelado.tif
- `nefinject inspect-nef`: Inspección detallada de IFDs, etiquetas TIFF y MakerNote Nikon.
- `nefinject characterize-sensor`: Estimación de ruido de lectura, ganancia (e-/DN), mapas PRNU y DSNU a partir de darks y flats.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .calibration import compare_bayer_syntheses
from .color import DngAdobeProfileExtractor, NefCameraColorProbe
from .container import NefInspector, NikonMakerNoteInspector, TiffReader
from .models import InjectionConfig, NefInjectError, sha256_file, write_json_atomic
from .optics import OpticsConfig, simulate_optics
from .pipeline import analyze_rendered_tiff, inject_synthetic_nef
from .sensor import NefSensorSimulator, SensorConfig, characterize_sensor


def _parse_wb(wb_str: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in wb_str.split(",")]
    if len(parts) != 4:
        raise ValueError("WB requiere cuatro ganancias R,G1,G2,B separadas por coma.")
    vals = tuple(float(x) for x in parts)
    if any(v <= 0 or v > 16 for v in vals):
        raise ValueError(f"Ganancias WB fuera de rango (0, 16]: {vals}")
    return vals  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nefinject",
        description="Librería científica de síntesis Bayer e inyección en archivos RAW Nikon NEF",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcomando principal: run
    p_run = sub.add_parser("run", help="Ejecuta el pipeline completo de síntesis e inyección NEF")
    p_run.add_argument("--nef", type=Path, required=True, help="Archivo .NEF original (plantilla)")
    p_run.add_argument("--dng", type=Path, default=None, help="Archivo .DNG para extraer perfil Adobe")
    p_run.add_argument("--graft", type=Path, default=None, help="Imagen RGB de entrada (anula --pattern)")
    p_run.add_argument(
        "--pattern",
        choices=["colorchecker", "star"],
        default="colorchecker",
        help="Patrón autogenerado sin --graft: colorchecker (predeterminado) o estrella Siemens (star)",
    )
    p_run.add_argument("--wb", type=str, default="1,1,1,1", help="Ganancias CFA R,G1,G2,B separadas por coma")
    p_run.add_argument("--ev", "--exposure-ev", dest="exposure_ev", type=float, default=-1.0, help="Compensación EV")
    p_run.add_argument("--kelvin", type=float, default=6200.0, help="Temperatura Kelvin de interpolación DNG")
    p_run.add_argument("--black-dn", type=int, default=0, help="Nivel de negro DN")
    p_run.add_argument("--white-dn", type=int, default=15000, help="Nivel de blanco DN")
    p_run.add_argument("--resize", choices=["contain", "cover", "stretch", "none"], default="cover")
    p_run.add_argument(
        "--orientation",
        choices=["preserve", "match-sensor"],
        default="preserve",
        help="Conservar la orientaci\u00f3n de la carta o ajustarla al sensor",
    )
    p_run.add_argument(
        "--rotation",
        choices=["cw", "ccw"],
        default=None,
        help="Sentido de giro si --orientation match-sensor detecta orientaciones distintas",
    )
    p_run.add_argument("--input-encoding", choices=["srgb", "linear", "auto"], default="srgb")
    p_run.add_argument("--strategy", choices=["replace-tail", "append"], default="replace-tail")
    p_run.add_argument("--out-dir", type=Path, default=Path("experimentos_dng"))
    p_run.add_argument("--run-dir", type=Path, default=None, help="Carpeta exacta de salida (opcional)")
    p_run.add_argument("--verification-mode", choices=["fast", "paranoid", "full", "none"], default="fast")
    p_run.add_argument("--codec-backend", choices=["auto", "numba", "numpy", "python"], default="auto")
    p_run.add_argument("--render-png", action="store_true", help="Generar validacion_rawpy_srgb8.png")
    p_run.add_argument("--allow-different-name", action="store_true")
    p_run.add_argument(
        "--optics-profile",
        type=str,
        default=None,
        help="Perfil óptico pre-CFA: ruta explícita a un archivo JSON",
    )
    p_run.add_argument(
        "--sensor-profile",
        type=str,
        default=None,
        help="Perfil de sensor post-CFA: ruta explícita a un archivo JSON",
    )
    p_run.add_argument(
        "--save-optics-tiff",
        action="store_true",
        help="Guardar en disco optica_pre_cfa_linear.tif (~294 MB) además de procesarlo en RAM",
    )

    # Subcomando: analyze-tiff
    p_tiff = sub.add_parser("analyze-tiff", help="Analiza cuantitativamente (ΔE00) un TIFF revelado")
    p_tiff.add_argument("--run-dir", type=Path, required=True, help="Carpeta del experimento generado")
    p_tiff.add_argument("--tiff", type=Path, required=True, help="Archivo TIFF revelado")
    p_tiff.add_argument("--layout", type=Path, default=None, help="Layout JSON personalizado")
    p_tiff.add_argument("--encoding", choices=["srgb", "linear"], default="srgb")
    p_tiff.add_argument("--ignore-icc", action="store_true")

    # Subcomando: inspect
    p_insp = sub.add_parser("inspect", help="Inspecciona un archivo NEF (TIFF + MakerNote + Probe)")
    p_insp.add_argument("nef", type=Path)
    p_insp.add_argument("--out-prefix", type=Path, default=None)

    # Subcomando: extract-dng
    p_dng = sub.add_parser("extract-dng", help="Extrae matrices y LUTs de un archivo DNG")
    p_dng.add_argument("dng", type=Path)
    p_dng.add_argument("--out-prefix", type=Path, default=None)
    p_dng.add_argument("--kelvin", type=float, default=6200.0)

    # Subcomando: compare
    p_cmp = sub.add_parser("compare", help="Compara dos mosaicos Bayer .npy y sus NEF")
    p_cmp.add_argument("--base", type=Path, required=True)
    p_cmp.add_argument("--prueba", type=Path, required=True)
    p_cmp.add_argument("--nef-base", type=Path, default=None)
    p_cmp.add_argument("--nef-prueba", type=Path, default=None)

    # Subcomando: simulate-optics
    p_opt = sub.add_parser("simulate-optics", help="Simula óptica pre-CFA y exporta TIFF RGB lineal float32")
    p_opt.add_argument("--input", type=Path, required=True, help="Imagen RGB de entrada")
    p_opt.add_argument("--config", type=str, required=True, help="Ruta explícita a un archivo JSON")
    p_opt.add_argument("--output", type=Path, required=True, help="Archivo TIFF float32 de salida")
    p_opt_grp = p_opt.add_mutually_exclusive_group(required=True)
    p_opt_grp.add_argument("--inspection", type=Path, help="inspection.json del NEF")
    p_opt_grp.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    p_opt.add_argument("--resize", choices=["none", "contain", "cover", "stretch"], default="contain")
    p_opt.add_argument("--input-encoding", choices=["srgb", "linear", "auto"], default="srgb")
    p_opt.add_argument("--ignore-icc", action="store_true")

    # Subcomando: simulate-sensor
    p_sen = sub.add_parser("simulate-sensor", help="Simula artefactos de sensor post-CFA sobre un Bayer .npy")
    p_sen.add_argument("--input", type=Path, required=True, help="Bayer ideal .npy uint16")
    p_sen.add_argument("--config", type=str, required=True, help="Ruta explícita a un archivo JSON")
    p_sen.add_argument("--output", type=Path, required=True, help="Bayer degradado .npy uint16")
    p_sen.add_argument("--source-manifest", type=Path, default=None)
    p_sen.add_argument("--overwrite", action="store_true")

    # Subcomando: characterize-sensor
    p_char = sub.add_parser("characterize-sensor", help="Caracteriza PRNU, DSNU, defectos y ganancia PTC desde darks/flats .npy")
    p_char.add_argument("--darks", required=True, nargs="+", type=Path)
    p_char.add_argument("--flats", required=True, nargs="+", type=Path)
    p_char.add_argument("--out-prefix", required=True, type=Path)
    p_char.add_argument("--cfa", default="RGGB", choices=["RGGB", "BGGR", "GRBG", "GBRG"])
    p_char.add_argument("--blur-sigma-px", type=float, default=35.0)
    p_char.add_argument("--black-dn", type=int, default=0)
    p_char.add_argument("--white-dn", type=int, default=15000)
    p_char.add_argument("--electrons-per-dn", type=float, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            wb_gains = _parse_wb(args.wb)
            cfg = InjectionConfig(
                nef_path=args.nef,
                dng_path=args.dng,
                chart_path=args.graft,
                pattern=args.pattern,
                output_dir=args.out_dir,
                run_dir=args.run_dir,
                wb_gains=wb_gains,
                exposure_ev=args.exposure_ev,
                kelvin=args.kelvin,
                black_dn=args.black_dn,
                white_dn=args.white_dn,
                resize_mode=args.resize,
                orientation=args.orientation,
                rotation=args.rotation,
                input_encoding=args.input_encoding,
                strategy=args.strategy,
                verification_mode=args.verification_mode,
                codec_backend=args.codec_backend,
                render_validation_png=args.render_png,
                allow_different_original_name=args.allow_different_name,
                optics_profile=args.optics_profile,
                sensor_profile=args.sensor_profile,
                save_optics_tiff=args.save_optics_tiff,
            )
            result = inject_synthetic_nef(cfg)
            print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "analyze-tiff":
            report_path, report = analyze_rendered_tiff(
                args.run_dir,
                args.tiff,
                custom_layout_path=args.layout,
                encoding=args.encoding,
                ignore_icc=args.ignore_icc,
            )
            print(
                json.dumps(
                    {
                        "report_path": str(report_path),
                        "summary": report["summary"],
                        "white_balance": report["white_balance"]["status"],
                        "exposure": report["exposure"]["status"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        if args.command == "inspect":
            prefix = args.out_prefix or args.nef.with_suffix("")
            sha = sha256_file(args.nef)
            reader = TiffReader(args.nef).parse()
            insp = NefInspector(args.nef, compute_sha256=True, precomputed_sha256=sha, reader=reader).inspect()
            mk = NikonMakerNoteInspector(
                args.nef, inspection=insp, compute_sha256=True, precomputed_sha256=sha, reader=reader
            ).inspect()
            probe = NefCameraColorProbe().inspect(
                args.nef, compute_sha256=True, precomputed_sha256=sha, makernote_report=mk
            )
            write_json_atomic(Path(f"{prefix}.inspection.json"), insp, overwrite=True)
            write_json_atomic(Path(f"{prefix}.makernote.json"), mk, overwrite=True)
            write_json_atomic(Path(f"{prefix}.color_probe.json"), probe, overwrite=True)
            print(f"Inspección completada: {prefix}.[inspection|makernote|color_probe].json")
            return 0

        if args.command == "extract-dng":
            prefix = args.out_prefix or args.dng.with_suffix("")
            extractor = DngAdobeProfileExtractor(args.dng)
            report, matrix, _ = extractor.extract_to_files(prefix, kelvin=args.kelvin)
            print(f"Perfil DNG extraído: {report['embedded_profile']['profile_name']}")
            return 0

        if args.command == "compare":
            res = compare_bayer_syntheses(
                args.base, args.prueba, nef_base=args.nef_base, nef_test=args.nef_prueba
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0

        if args.command == "simulate-optics":
            if args.inspection:
                data = json.loads(args.inspection.read_text(encoding="utf-8"))["raw"]
                width, height = int(data["width"]), int(data["height"])
            else:
                width, height = args.size
            opt_cfg, _ = OpticsConfig.load(args.config)
            rep = simulate_optics(
                args.input,
                args.output,
                opt_cfg,
                width=width,
                height=height,
                resize=args.resize,
                encoding=args.input_encoding,
                ignore_icc=args.ignore_icc,
            )
            print(json.dumps(rep["output"], indent=2, ensure_ascii=False))
            return 0

        if args.command == "simulate-sensor":
            sen_cfg, _ = SensorConfig.load(args.config)
            rep = NefSensorSimulator(sen_cfg).simulate(
                args.input,
                args.output,
                source_manifest=args.source_manifest,
                overwrite=args.overwrite,
            )
            print(json.dumps(rep["quality"], indent=2, ensure_ascii=False))
            return 0

        if args.command == "characterize-sensor":
            rep = characterize_sensor(
                args.darks,
                args.flats,
                args.out_prefix,
                cfa=args.cfa,
                blur_sigma_px=args.blur_sigma_px,
                black_dn=args.black_dn,
                white_dn=args.white_dn,
                assumed_e_per_dn=args.electrons_per_dn,
            )
            print(json.dumps(rep["estimates"], indent=2, ensure_ascii=False))
            return 0

    except (NefInjectError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

