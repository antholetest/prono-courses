# utils/database.py
import json
import logging
from pathlib import Path
import sqlite3
import streamlit as st

logger = logging.getLogger("PMU_Pro")

DOSSIER = Path(".")
DB_PATH = DOSSIER / "pmu_database.db"

def get_connection():
    url = st.secrets.get("TURSO_DATABASE_URL")
    token = st.secrets.get("TURSO_AUTH_TOKEN")
    
    if not url or not token:
        logger.error("❌ TURSO : Identifiants introuvables dans st.secrets !")
        return sqlite3.connect(DB_PATH)
    
    try:
        import libsql
        conn = libsql.connect(database=url, auth_token=token)
        logger.info("✅ TURSO : Connexion Cloud réussie !")
        return conn
    except Exception as e:
        logger.error(f"❌ ERREUR TURSO EXACTE : {e}")
        return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paris (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            reunion TEXT,
            course_num TEXT,
            course TEXT,
            discipline TEXT,
            type TEXT,
            details TEXT,
            mise REAL,
            statut TEXT,
            gain REAL,
            diagnostic TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS modele_ia (
            cle TEXT PRIMARY KEY,
            valeur TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses_cache (
            date_iso TEXT,
            reunion TEXT,
            course TEXT,
            data_json TEXT,
            PRIMARY KEY (date_iso, reunion, course)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bilans_journee (
            date_iso TEXT,
            reunion_hippodrome TEXT,
            data_json TEXT,
            PRIMARY KEY (date_iso, reunion_hippodrome)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paris_date ON paris(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paris_statut ON paris(statut)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_courses_cache_date ON courses_cache(date_iso)")
    
    conn.commit()
    conn.close()
    logger.info("Base de données initialisée avec succès.")

def migrer_anciens_json_vers_sqlite():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Migration historique_paris.json
    f_hist = DOSSIER / "historique_paris.json"
    if f_hist.exists():
        cursor.execute("SELECT COUNT(*) FROM paris")
        first_row = cursor.fetchone()
        count_val = first_row[0] if isinstance(first_row, (tuple, list)) else first_row["COUNT(*)"] if isinstance(first_row, dict) else 0
        
        if count_val == 0:
            try:
                with open(f_hist, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for p in data:
                        try:
                            cursor.execute("""
                                INSERT INTO paris (date, reunion, course_num, course, discipline, type, details, mise, statut, gain, diagnostic)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                p.get("date"), p.get("reunion"), p.get("course_num"), p.get("course"),
                                p.get("discipline"), p.get("type"), p.get("details"),
                                p.get("mise", 0.0), p.get("statut", "En attente"),
                                p.get("gain", 0.0), p.get("diagnostic", "")
                            ))
                        except Exception as e_pari:
                            logger.error(f"Erreur insertion pari individuel : {e_pari}")
                conn.commit()
                logger.info("Migration de l'historique des paris réussie.")
            except Exception as e:
                logger.error(f"Erreur migration historique globale : {e}")

    # 2. Migration modele_ia_pmu.json
    f_modele = DOSSIER / "modele_ia_pmu.json"
    if f_modele.exists():
        cursor.execute("SELECT COUNT(*) FROM modele_ia")
        first_row = cursor.fetchone()
        count_val = first_row[0] if isinstance(first_row, (tuple, list)) else first_row["COUNT(*)"] if isinstance(first_row, dict) else 0
        
        if count_val == 0:
            try:
                with open(f_modele, "r", encoding="utf-8") as f:
                    modele_data = json.load(f)
                    for k, v in modele_data.items():
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO modele_ia (cle, valeur) VALUES (?, ?)
                            """, (k, json.dumps(v)))
                        except Exception as e_m:
                            logger.error(f"Erreur insertion clé modèle IA {k}: {e_m}")
                conn.commit()
                logger.info("Migration du modèle IA réussie.")
            except Exception as e:
                logger.error(f"Erreur migration modèle IA global : {e}")

    # 3. Migration pmu_du_jour_*.json
    for f_json in DOSSIER.glob("pmu_du_jour_*.json"):
        try:
            date_iso = f_json.stem.replace("pmu_du_jour_", "")
            with open(f_json, "r", encoding="utf-8") as f:
                courses = json.load(f)
                for course in courses:
                    reunion = course.get("reunion", "R?")
                    course_num = course.get("course", "C?")
                    cursor.execute("""
                        INSERT OR REPLACE INTO courses_cache (date_iso, reunion, course, data_json)
                        VALUES (?, ?, ?, ?)
                    """, (date_iso, reunion, course_num, json.dumps(course, ensure_ascii=False)))
            conn.commit()
        except Exception as e:
            logger.error(f"Erreur migration courses {f_json.name}: {e}")

    # 4. Migration bilan_journee_*.json
    for f_json in DOSSIER.glob("bilan_journee_*.json"):
        try:
            date_iso = f_json.stem.replace("bilan_journee_", "")
            with open(f_json, "r", encoding="utf-8") as f:
                bilan_data = json.load(f)
                for item in bilan_data:
                    reunion_hyp = item.get("Réunion / Hippodrome", "Inconnu")
                    cursor.execute("""
                        INSERT OR REPLACE INTO bilans_journee (date_iso, reunion_hippodrome, data_json)
                        VALUES (?, ?, ?)
                    """, (date_iso, reunion_hyp, json.dumps(item, ensure_ascii=False)))
            conn.commit()
        except Exception as e:
            logger.error(f"Erreur migration bilan {f_json.name}: {e}")

    conn.close()

def charger_historique():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, date, reunion, course_num, course, discipline, type, details, mise, statut, gain, diagnostic FROM paris")
    rows = cursor.fetchall()
    conn.close()
    
    colonnes = ["id", "date", "reunion", "course_num", "course", "discipline", "type", "details", "mise", "statut", "gain", "diagnostic"]
    
    resultats = []
    for row in rows:
        if isinstance(row, dict):
            resultats.append(row)
        else:
            resultats.append(dict(zip(colonnes, row)))
    return resultats

def sauvegarder_historique(historique):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM paris")
    for p in historique:
        cursor.execute("""
            INSERT INTO paris (date, reunion, course_num, course, discipline, type, details, mise, statut, gain, diagnostic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.get("date"), p.get("reunion"), p.get("course_num"), p.get("course"),
            p.get("discipline"), p.get("type"), p.get("details"),
            p.get("mise", 0.0), p.get("statut", "En attente"),
            p.get("gain", 0.0), p.get("diagnostic", "")
        ))
    conn.commit()
    conn.close()

def charger_courses_jour_db(date_iso):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT data_json FROM courses_cache WHERE date_iso = ?", (date_iso,))
    rows = cursor.fetchall()
    conn.close()
    
    donnees = []
    for row in rows:
        valeur_json = row["data_json"] if isinstance(row, dict) else row[0]
        donnees.append(json.loads(valeur_json))
        
    reunions_map = {}
    for elem in donnees:
        cle = f"{elem.get('reunion', 'R?')} - {elem.get('hippodrome', 'Hippodrome')}"
        if cle not in reunions_map:
            reunions_map[cle] = []
        reunions_map[cle].append(elem)
    return donnees, reunions_map

def sauvegarder_courses_jour_db(date_iso, resultat_journee):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM courses_cache WHERE date_iso = ?", (date_iso,))
    for course in resultat_journee:
        reunion = course.get("reunion", "R?")
        course_num = course.get("course", "C?")
        cursor.execute("""
            INSERT OR REPLACE INTO courses_cache (date_iso, reunion, course, data_json)
            VALUES (?, ?, ?, ?)
        """, (date_iso, reunion, course_num, json.dumps(course, ensure_ascii=False)))
    conn.commit()
    conn.close()