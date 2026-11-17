"""Perfiles JSON de simulación óptica y de sensor para nefinject.

Forma parte de la librería `nefinject`, centralizando las configuraciones JSON de
referencia para el modelado óptico (PSF, OLPF, viñeteado, aberración cromática)
y el modelado de sensor (ruido de lectura, FPN, PRNU, defectos de fotositos).

Los perfiles se cargan proporcionando una ruta explícita a su archivo JSON. Este
módulo solo expone la ubicación del directorio y de su documentación.

Author: Jose Pereira (http://www.jpereira.net/)
Copyright: (c) Jose Pereira - http://www.jpereira.net/
"""
from __future__ import annotations

from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent

README_PATH = PROFILES_DIR / "README_SIMULADORES.md"

