# utils/database.py
import json
import logging
from pathlib import Path
import sqlite3
import streamlit as st

logger = logging.getLogger("PMU_Pro")

DOSSIER = Path(".")
DB_PATH = DOSSIER / "pmu_database.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
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
        
        # Ajout d'index pour optimiser les performances de recherche et de filtrage
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paris_date ON paris(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paris_statut ON paris(statut)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_courses_cache_date ON courses_cache(date_iso)")
        
        conn.commit()
    logger.info("Base de données SQLite initialisée avec succès (avec index de performance).")

def migrer_anciens_json_vers_sqlite():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # 1. Migration historique_paris.json
        f_hist = DOSSIER / "historique_paris.json"
        if f_hist.exists():
            cursor.execute("SELECT COUNT(*) FROM paris")
            if cursor.fetchone()[0] == 0:
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
                    # Archivage sécurisé pour éviter de refaire la migration
                    f_hist.replace(f_hist.with_name(f_hist.name + ".bak"))
                    logger.info("Migration de l'historique des paris réussie. Fichier archivé en .bak")
                except Exception as e:
                    logger.error(f"Erreur migration historique globale : {e}")

        # 2. Migration modele_ia_pmu.json
        f_modele = DOSSIER / "modele_ia_pmu.json"
        if f_modele.exists():
            cursor.execute("SELECT COUNT(*) FROM modele_ia")
            if cursor.fetchone()[0] == 0:
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
                    f_modele.replace(f_modele.with_name(f_modele.name + ".bak"))
                    logger.info("Migration du modèle IA réussie. Fichier archivé en .bak")
                except Exception as e:
                    logger.error(f"Erreur migration modèle IA global : {e}")

        # 3. Migration pmu_du_jour_*.json
        for f_json in list(DOSSIER.glob("pmu_du_jour_*.json")):
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
                f_json.replace(f_json.with_name(f_json.name + ".bak"))
            except Exception as e:
                logger.error(f"Erreur migration courses {f_json.name}: {e}")

        # 4. Migration bilan_journee_*.json
        for f_json in list(DOSSIER.glob("bilan_journee_*.json")):
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
                f_json.replace(f_json.with_name(f_json.name + ".bak"))
            except Exception as e:
                logger.error(f"Erreur migration bilan {f_json.name}: {e}")

def charger_historique():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM paris")
        # Les dictionnaires générés incluront désormais l'identifiant unique "id"
        rows = [dict(row) for row in cursor.fetchall()]
    return rows

def sauvegarder_historique(historique):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # 1. Identifier les IDs actuellement présents dans la liste envoyée
        ids_fournis = [p["id"] for p in historique if "id" in p]
        
        # 2. Supprimer de la base de données UNIQUEMENT les paris qui ont été retirés de la liste
        if ids_fournis:
            placeholders = ",".join("?" * len(ids_fournis))
            cursor.execute(f"DELETE FROM paris WHERE id NOT IN ({placeholders})", ids_fournis)
        else:
            # Si la liste est vide, on vide la table
            cursor.execute("DELETE FROM paris")
            
        # 3. Mettre à jour les paris existants ou insérer les nouveaux
        for p in historique:
            if "id" in p:
                cursor.execute("""
                    UPDATE paris SET 
                        date=?, reunion=?, course_num=?, course=?, discipline=?, 
                        type=?, details=?, mise=?, statut=?, gain=?, diagnostic=?
                    WHERE id=?
                """, (
                    p.get("date"), p.get("reunion"), p.get("course_num"), p.get("course"),
                    p.get("discipline"), p.get("type"), p.get("details"),
                    p.get("mise", 0.0), p.get("statut", "En attente"),
                    p.get("gain", 0.0), p.get("diagnostic", ""),
                    p["id"]  # Clé unique pour la mise à jour
                ))
            else:
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

def charger_courses_jour_db(date_iso):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM courses_cache WHERE date_iso = ?", (date_iso,))
        rows = cursor.fetchall()
    
    donnees = [json.loads(row[0]) for row in rows]
    reunions_map = {}
    for elem in donnees:
        cle = f"{elem.get('reunion', 'R?')} - {elem.get('hippodrome', 'Hippodrome')}"
        if cle not in reunions_map:
            reunions_map[cle] = []
        reunions_map[cle].append(elem)
    return donnees, reunions_map

def sauvegarder_courses_jour_db(date_iso, resultat_journee):
    with sqlite3.connect(DB_PATH) as conn:
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