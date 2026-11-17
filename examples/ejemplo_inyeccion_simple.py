r"""Ejemplo 1: Ejecución directa del pipeline completo de inyección sintética NEF desde Python.

Equivalente nativo en Python al comando de PowerShell:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\generar_nef_perfil_dng.ps1" \
      -Nef C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.NEF \
      -Dng C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.DNG \
      -Carta C:\Users\jose\PycharmProjects\fake_nef\nef_synthetic_injector_etapas_1_7_clean\carta_color_demo.png \
      -WB "1,1,1,1" -ExposicionEV -1.00 -Kelvin 6200 -BlackDN 0 -WhiteDN 15000
"""
from __future__ import annotations

import json
from pathlib import Path

from nefinject import inject_synthetic_nef


def main() -> None:
    result = inject_synthetic_nef(
        nef_path=r"C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.NEF",
        dng_path=r"C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.DNG",
        chart_path=r"C:\Users\jose\PycharmProjects\fake_nef\nef_synthetic_injector_etapas_1_7_clean\carta_color_demo.png",
        output_dir=Path(__file__).resolve().parent.parent / "experimentos_dng",
        wb_gains=(1.0, 1.0, 1.0, 1.0),
        exposure_ev=-1.00,
        kelvin=6200.0,
        black_dn=0,
        white_dn=15000,
        verification_mode="fast",  # Cambiar a "paranoid" para activar también check_codec completo + LibRaw C probe
    )

    print("=== NEF SINTÉTICO GENERADO Y VALIDADO ===")
    print(f"Carpeta de salida : {result.run_dir}")
    print(f"Archivo NEF       : {result.output_nef_path}")
    print(f"SHA-256 NEF       : {result.output_nef_sha256}")
    print(f"SHA-256 Bayer NPY : {result.bayer_sha256}")
    print(f"SHA-256 Payload   : {result.compressed_payload_sha256}")
    print("\nDesglose de tiempos (ms):")
    print(json.dumps(result.timings_ms, indent=2))


if __name__ == "__main__":
    main()
