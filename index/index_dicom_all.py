#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_dicom_all.py — Indexador combinado de estudios DICOM
===========================================================

Orquesta los indexadores de PET/CT y MRI para generar una base de datos
y un árbol JSON unificados que contienen toda la información de los
estudios DICOM.

Genera:
  - dicom_all_index.db  : SQLite combinada con tablas prefijadas (pet_ct_*, mri_*)
  - dicom_all_tree.json : Árbol JSON unificado con ambas modalidades

Uso (a traves de larmornium.py):
    python3 larmornium.py index --dicom-dir ./DICOM
    python3 larmornium.py index --dicom-dir ./DICOM --output-dir ./output --verbose

Internamente importa y ejecuta:
  - index_pet_ct.index_pet_ct() -> pet_ct_index.db + pet_ct_tree.json
  - index_mri.index_mri()       -> mri_index.db   + mri_tree.json
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

# Importar los indexadores individuales como módulos
import index_pet_ct
import index_mri

# Logging
logger = logging.getLogger("index_dicom_all")

# Funciones de combinación
def copy_tables_with_prefix(src_db_path, dst_conn, prefix):
    """
    Copia todas las tablas de una base de datos SQLite de origen
    a la conexión destino, añadiendo un prefijo a cada nombre de tabla.

    Parameters
    ----------
    src_db_path : str
        Ruta a la base de datos de origen.
    dst_conn : sqlite3.Connection
        Conexión a la base de datos de destino.
    prefix : str
        Prefijo a añadir a cada tabla (e.g., 'pet_ct_', 'mri_').
    """
    src_conn = sqlite3.connect(src_db_path)
    src_cursor = src_conn.cursor()
    dst_cursor = dst_conn.cursor()

    # Obtener lista de tablas
    src_cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = src_cursor.fetchall()

    for table_name, create_sql in tables:
        new_table_name = f"{prefix}{table_name}"

        # Reemplazar el nombre de la tabla en el CREATE statement
        # El CREATE SQL tiene formato: CREATE TABLE IF NOT EXISTS nombre (...)
        # o CREATE TABLE nombre (...)
        new_create_sql = create_sql.replace(
            f"CREATE TABLE IF NOT EXISTS {table_name}",
            f"CREATE TABLE IF NOT EXISTS {new_table_name}",
            1
        )
        if new_create_sql == create_sql:
            # Intentar sin IF NOT EXISTS
            new_create_sql = create_sql.replace(
                f"CREATE TABLE {table_name}",
                f"CREATE TABLE IF NOT EXISTS {new_table_name}",
                1
            )

        # También actualizar las FOREIGN KEY references con el prefijo
        # Esto es necesario para que las FK apunten a las tablas prefijadas
        for other_name, _ in tables:
            if other_name != table_name:
                new_create_sql = new_create_sql.replace(
                    f"REFERENCES {other_name}(",
                    f"REFERENCES {prefix}{other_name}("
                )

        # Crear la tabla en destino
        try:
            dst_cursor.execute(new_create_sql)
        except sqlite3.OperationalError as e:
            logger.warning("Error creando tabla %s: %s", new_table_name, e)
            continue

        # Copiar datos
        src_cursor.execute(f"SELECT * FROM {table_name}")
        rows = src_cursor.fetchall()

        if rows:
            # Obtener nombres de columnas
            src_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in src_cursor.fetchall()]
            placeholders = ", ".join(["?"] * len(columns))
            cols_str = ", ".join(columns)

            insert_sql = f"INSERT OR IGNORE INTO {new_table_name} ({cols_str}) VALUES ({placeholders})"
            dst_cursor.executemany(insert_sql, rows)

        logger.info("  Tabla %s: %d registros copiados", new_table_name, len(rows))

    src_conn.close()
    dst_conn.commit()


def create_summary_table(dst_conn, pet_ct_db, mri_db):
    """
    Crea una tabla de resumen con estadísticas globales.
    """
    dst_cursor = dst_conn.cursor()

    dst_cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            key    TEXT PRIMARY KEY,
            value  TEXT
        )
    """)

    stats = {}

    # Contar registros de PET/CT
    if os.path.exists(pet_ct_db):
        pc_conn = sqlite3.connect(pet_ct_db)
        pc_cur = pc_conn.cursor()

        for table in ["studies", "series", "images", "ct_parameters",
                       "pet_parameters", "radiopharmaceutical_info",
                       "fusion_pairs"]:
            try:
                pc_cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"pet_ct_{table}_count"] = str(pc_cur.fetchone()[0])
            except sqlite3.OperationalError:
                stats[f"pet_ct_{table}_count"] = "0"

        # Modalidades
        try:
            pc_cur.execute("SELECT DISTINCT modality FROM series")
            modalities = [r[0] for r in pc_cur.fetchall()]
            stats["pet_ct_modalities"] = ", ".join(modalities)
        except sqlite3.OperationalError:
            pass

        pc_conn.close()

    # Contar registros de MRI
    if os.path.exists(mri_db):
        mr_conn = sqlite3.connect(mri_db)
        mr_cur = mr_conn.cursor()

        for table in ["studies", "series", "images", "mr_parameters",
                       "analyze_volumes"]:
            try:
                mr_cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"mri_{table}_count"] = str(mr_cur.fetchone()[0])
            except sqlite3.OperationalError:
                stats[f"mri_{table}_count"] = "0"

        # File types
        try:
            mr_cur.execute("SELECT DISTINCT file_type FROM images")
            ftypes = [r[0] for r in mr_cur.fetchall()]
            stats["mri_file_types"] = ", ".join(ftypes)
        except sqlite3.OperationalError:
            pass

        mr_conn.close()

    stats["scan_date"] = datetime.now().isoformat()

    for key, value in stats.items():
        dst_cursor.execute(
            "INSERT OR REPLACE INTO summary (key, value) VALUES (?, ?)",
            (key, value)
        )

    dst_conn.commit()


def merge_json_trees(pet_ct_json_path, mri_json_path, output_path):
    """
    Combina los árboles JSON de PET/CT y MRI en un solo archivo.
    """
    pet_ct_tree = {}
    mri_tree = {}

    if os.path.exists(pet_ct_json_path):
        with open(pet_ct_json_path, "r", encoding="utf-8") as f:
            pet_ct_tree = json.load(f)

    if os.path.exists(mri_json_path):
        with open(mri_json_path, "r", encoding="utf-8") as f:
            mri_tree = json.load(f)

    combined = {
        "root": "DICOM",
        "scan_date": datetime.now().isoformat(),
        "description": "Índice combinado de estudios DICOM PET/CT y MRI",
        "summary": {
            "pet_ct": {
                "total_files": pet_ct_tree.get("total_files", 0),
                "total_dicom_files": pet_ct_tree.get("total_dicom_files", 0),
                "total_studies": pet_ct_tree.get("total_studies", 0),
                "total_multi_studies": pet_ct_tree.get("total_multi_studies", 0),
                "total_series": pet_ct_tree.get("total_series", 0),
            },
            "mri": {
                "total_files": mri_tree.get("total_files", 0),
                "total_medical_files": mri_tree.get("total_medical_files", 0),
                "total_studies": mri_tree.get("total_studies", 0),
                "total_multi_studies": mri_tree.get("total_multi_studies", 0),
                "total_series": mri_tree.get("total_series", 0),
                "file_type_counts": mri_tree.get("file_type_counts", {}),
            },
        },
        "modalities": {
            "PET_CT": pet_ct_tree,
            "MRI": mri_tree,
        },
    }

    all_patients = []
    for p in pet_ct_tree.get("patients", []):
        p_copy = dict(p)
        p_copy["modality"] = "PET_CT"
        all_patients.append(p_copy)
    for p in mri_tree.get("patients", []):
        p_copy = dict(p)
        p_copy["modality"] = "MRI"
        all_patients.append(p_copy)
    all_patients.sort(key=lambda x: str(x.get("patient_id", "")))

    combined["patients"] = all_patients

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return output_path


# Función principal
def index_all(dicom_dir, output_dir=None, verbose=False):
    """
    Ejecuta ambos indexadores y combina los resultados.

    Parameters
    ----------
    dicom_dir : str
        Ruta al directorio raíz DICOM.
    output_dir : str, optional
        Directorio de salida para los archivos generados.
    verbose : bool
        Progreso detallado.

    Returns
    -------
    tuple(str, str)
        Rutas de (dicom_all_index.db, dicom_all_tree.json).
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Paso 1: Ejecutar indexador PET/CT
    logger.info("=" * 60)
    logger.info("PASO 1/3: Indexando estudios PET/CT...")
    logger.info("=" * 60)
    pet_ct_db = None
    pet_ct_json = None
    try:
        pet_ct_db, pet_ct_json = index_pet_ct.index_pet_ct(
            dicom_dir, output_dir, verbose
        )
    except FileNotFoundError:
        logger.warning("No se encontro carpeta PET_CT, continuando...")
    except Exception as e:
        logger.error("Error indexando PET/CT: %s", e)

    # Paso 2: Ejecutar indexador MRI
    logger.info("")
    logger.info("=" * 60)
    logger.info("PASO 2/3: Indexando estudios MRI...")
    logger.info("=" * 60)
    mri_db = None
    mri_json = None
    try:
        mri_db, mri_json = index_mri.index_mri(
            dicom_dir, output_dir, verbose
        )
    except FileNotFoundError:
        logger.warning("No se encontro carpeta MRI, continuando...")
    except Exception as e:
        logger.error("Error indexando MRI: %s", e)

    # Paso 3: Combinar resultados
    logger.info("")
    logger.info("=" * 60)
    logger.info("PASO 3/3: Combinando resultados...")
    logger.info("=" * 60)

    combined_db_path = os.path.join(output_dir, "dicom_all_index.db")
    combined_json_path = os.path.join(output_dir, "dicom_all_tree.json")

    # Crear DB combinada
    if os.path.exists(combined_db_path):
        os.remove(combined_db_path)

    combined_conn = sqlite3.connect(combined_db_path)

    # Copiar tablas con prefijo
    if pet_ct_db and os.path.exists(pet_ct_db):
        logger.info("Copiando tablas PET/CT con prefijo 'pet_ct_'...")
        copy_tables_with_prefix(pet_ct_db, combined_conn, "pet_ct_")

    if mri_db and os.path.exists(mri_db):
        logger.info("Copiando tablas MRI con prefijo 'mri_'...")
        copy_tables_with_prefix(mri_db, combined_conn, "mri_")

    # Tabla de resumen
    logger.info("Creando tabla de resumen...")
    create_summary_table(
        combined_conn,
        pet_ct_db or "",
        mri_db or ""
    )

    combined_conn.close()

    # Combinar JSONs
    logger.info("Combinando árboles JSON...")
    merge_json_trees(
        pet_ct_json or "",
        mri_json or "",
        combined_json_path
    )

    # Resumen final
    logger.info("")
    logger.info("=" * 60)
    logger.info("INDEXACIÓN COMPLETA FINALIZADA")
    logger.info("=" * 60)
    logger.info("Archivos generados:")
    logger.info("  PET/CT:")
    if pet_ct_db:
        logger.info("    DB   : %s", pet_ct_db)
        logger.info("    JSON : %s", pet_ct_json)
    else:
        logger.info("    (no disponible)")
    logger.info("  MRI:")
    if mri_db:
        logger.info("    DB   : %s", mri_db)
        logger.info("    JSON : %s", mri_json)
    else:
        logger.info("    (no disponible)")
    logger.info("  Combinado:")
    logger.info("    DB   : %s", combined_db_path)
    logger.info("    JSON : %s", combined_json_path)

    return combined_db_path, combined_json_path
