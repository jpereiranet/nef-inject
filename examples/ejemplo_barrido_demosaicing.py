"""Ejemplo 2: Barrido en lote de alta velocidad con NefSession.

Muestra cómo reutilizar la sesión en memoria (cabeceras TIFF, MakerNote, prefijo binario
de 1.7 MB y matrices DNG) para generar múltiples archivos NEF sintéticos variando
la exposición EV o la carta de entrada sin repetir lecturas de disco sobre la plantilla.
"""
from __future__ import annotations

import time
from pathlib import Path

from nefinject import NefSession


def main() -> None:
    nef_template = r"C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.NEF"
    dng_template = r"C:\Users\jose\PycharmProjects\fake_nef\DSC_4415.DNG"
    carta = r"C:\Users\jose\PycharmProjects\fake_nef\nef_synthetic_injector_etapas_1_7_clean\carta_color_demo.png"
    out_root = Path(__file__).resolve().parent.parent / "experimentos_dng" / "barrido_ev"

    t0 = time.perf_counter()
    session = NefSession(nef_template, dng_template, codec_backend="auto")
    print(f"Sesión inicializada en {(time.perf_counter() - t0) * 1000:.1f} ms")

    for ev in (-2.0, -1.0, -0.5, 0.0):
        res = session.inject(
            chart_path=carta,
            output_dir=out_root,
            wb_gains=(1.0, 1.0, 1.0, 1.0),
            exposure_ev=ev,
            kelvin=6200.0,
            black_dn=0,
            white_dn=15000,
            verification_mode="fast",
            persist_intermediates=False,  # Solo escribe el .NEF, .injection.json y resumen_ejecucion.json
        )
        print(
            f"  [EV={ev:+.1f}] -> {res.output_nef_path.name} "
            f"| SHA256={res.output_nef_sha256[:12]}... "
            f"| Tiempo={res.timings_ms['total_ms']:.1f} ms"
        )


if __name__ == "__main__":
    main()

