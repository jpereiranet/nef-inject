r"""Ejemplo 3: Inyección NEF con simulación de óptica pre-CFA y artefactos de sensor post-CFA.

Equivalente nativo en memoria en Python al comando de PowerShell:
    .\generar_nef_con_artefactos.ps1 `
        -Nef 'DSC_4415.NEF' -Dng 'DSC_4415.DNG' -Carta 'carta_color_demo.png' `
        -WB '1,1,1,1' -ExposicionEV -0.25 -Kelvin 6200 -BlackDN 0 -WhiteDN 15000 `
        -OpticsProfile '.\optics_profile_example.json' `
        -SensorProfile '.\sensor_profile_noise_only.json'
"""
from __future__ import annotations

import json
from pathlib import Path

from nefinject import inject_synthetic_nef


def main() -> None:
    profiles_dir = Path(__file__).resolve().parents[1] / "src" / "nefinject" / "profiles"
    result = inject_synthetic_nef(
        nef_path=r"C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.NEF",
        dng_path=r"C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.DNG",
        chart_path=r"C:\Users\jose\PycharmProjects\fake_nef\nef_synthetic_injector_etapas_1_7_clean\carta_color_demo.png",
        output_dir=Path(__file__).resolve().parent.parent / "experimentos_dng",
        wb_gains=(1.0, 1.0, 1.0, 1.0),
        exposure_ev=-0.25,
        kelvin=6200.0,
        black_dn=0,
        white_dn=15000,
        optics_profile=profiles_dir / "optics_profile_example.json",
        sensor_profile=profiles_dir / "sensor_profile_noise_only.json",
        save_optics_tiff=False,         # En RAM por defecto (no escribe el TIFF de 294 MB en disco)
        verification_mode="fast",
    )

    print("=== NEF CON ÓPTICA Y SENSOR SIMULADOS GENERADO ===")
    print(f"Carpeta de salida          : {result.run_dir}")
    print(f"Archivo NEF                : {result.output_nef_path}")
    print(f"SHA-256 Bayer Ideal        : {result.bayer_sha256}")
    print(f"SHA-256 Bayer Con Sensor   : {result.injected_bayer_sha256}")
    print(f"SHA-256 NEF Final          : {result.output_nef_sha256}")
    if result.sensor_manifest:
        frac = result.sensor_manifest["quality"]["changed_fraction"] * 100.0
        print(f"Fotositos modificados      : {frac:.2f}%")
    print("\nDesglose de tiempos (ms):")
    print(json.dumps(result.timings_ms, indent=2))


if __name__ == "__main__":
    main()

