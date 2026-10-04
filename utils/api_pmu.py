# utils/api_pmu.py
import datetime
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
from utils.database import DB_PATH, sauvegarder_courses_jour_db

logger = logging.getLogger("PMU_Pro")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

def safe_float(val, default=0.0):
    if val is None or val == "" or val == "-":
        return default
    try:
        return float(str(val).replace(",", ".").replace("€", "").strip())
    except (ValueError, TypeError):
        return default

def detecter_discipline(course_obj):
    api_disc = str(course_obj.get("discipline", "")).upper()
    api_spec = str(course_obj.get("specialite", "")).upper()
    combined = f"{api_disc} {api_spec}"

    if "ATTELE" in combined or "TROT_ATTELE" in combined:
        return "Trot Attelé"
    elif "MONTE" in combined or "TROT_MONTE" in combined:
        return "Trot Monté"
    elif "HAIES" in combined:
        return "Haies"
    elif "STEEPLE" in combined:
        return "Steeple-chase"
    elif "PLAT" in combined or "GALOP" in combined:
        return "Galop Plat"

    texte = f"{course_obj.get('libelle', '')} {course_obj.get('conditions', '')}".upper()
    if "MONTÉ" in texte or "MONTE" in texte:
        return "Trot Monté"
    elif "HAIES" in texte:
        return "Haies"
    elif "STEEPLE" in texte:
        return "Steeple-chase"
    elif "PLAT" in texte or "GALOP" in texte or "HANDICAP" in texte:
        return "Galop Plat"
    elif "ATTELÉ" in texte or "ATTELE" in texte or "TROT" in texte:
        return "Trot Attelé"
    return "Galop Plat"

def detecter_corde(nom_course, conditions_texte=""):
    texte = f"{nom_course} {conditions_texte}".upper()
    if "GAUCHE" in texte:
        return "Corde à gauche ↺"
    elif "DROITE" in texte:
        return "Corde à droite ↻"
    return "Corde standard"

def detecter_etat_terrain(conditions_texte):
    if not conditions_texte:
        return "Bon (Standard)"
    texte = str(conditions_texte).upper()
    if "LOURD" in texte:
        return "Lourd"
    elif "COLLANT" in texte:
        return "Collant"
    elif "SOUPLE" in texte:
        return (
            "Collant"
            if ("TRES SOUPLE" in texte or "TRÈS SOUPLE" in texte)
            else "Souple"
        )
    return "Bon (Standard)"

def telecharger_pmu_date(date_iso, fichier_cible=None, afficher_progres=True):
    try:
        dt = datetime.datetime.strptime(date_iso, "%Y-%m-%d")
        date_pmu = dt.strftime("%d%m%Y")
    except Exception as e:
        logger.warning(f"Format de date invalide pour l'API PMU ({date_iso}) : {e}")
        return False

    url_programme = (
        f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}"
    )
    
    # Utilisation d'une Session pour optimiser les performances réseau et ajouter une résilience
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Configuration d'une stratégie de reconnexion automatique (Retry)
    retry_strategy = Retry(
        total=3,  # 3 tentatives maximum
        backoff_factor=1,  # Temps d'attente croissant (1s, 2s, 4s) entre chaque tentative
        status_forcelist=[429, 500, 502, 503, 504],  # Codes HTTP déclenchant une relance
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        res = session.get(url_programme, timeout=15)
        if res.status_code != 200:
            logger.warning(f"Impossible de récupérer le programme PMU pour {date_iso} (Code HTTP: {res.status_code})")
            return False
        data = res.json()
    except Exception as e:
        logger.error(f"Erreur réseau lors de l'appel de l'API PMU pour la date {date_iso} : {e}")
        return False

    reunions = data.get("programme", {}).get("reunions", [])
    if not reunions:
        logger.info(f"Aucune réunion trouvée pour la date {date_iso}.")
        return False

    resultats_journee = []
    total_courses = sum(len(r.get("courses", [])) for r in reunions)
    courses_traitees = 0
    
    progress_bar = st.progress(0, text="Téléchargement des programmes en cours...") if afficher_progres else None

    for reunion in reunions:
        num_r = f"R{reunion.get('numOfficiel')}"
        hippodrome = reunion.get("hippodrome", {}).get("libelleLong", "")

        for course in reunion.get("courses", []):
            courses_traitees += 1
            if afficher_progres and progress_bar and total_courses > 0:
                prog_val = min(1.0, courses_traitees / total_courses)
                progress_bar.progress(prog_val, text=f"Téléchargement : {num_r} - Course {courses_traitees}/{total_courses}")

            num_c = f"C{course.get('numOrdre')}"
            nom_course = course.get("libelle", "")
            discipline = detecter_discipline(course)
            conditions_course = course.get("conditions", "")
            terrain_detecte = detecter_etat_terrain(conditions_course)
            corde_detectee = detecter_corde(nom_course, conditions_course)
            distance_val = course.get("distance") or course.get("distanceTotale", "")
            
            heure_str = "13:30"
            valeurs_a_tester = [
                course.get("dateTheoriqueDepart"),
                course.get("heureDepart"),
                course.get("pariHeureDepart"),
            ]
            for val in valeurs_a_tester:
                if val:
                    try:
                        if isinstance(val, (int, float)) and val > 100000:
                            diviseur = 1000.0 if val > 1e10 else 1.0
                            dt_utc = datetime.datetime.fromtimestamp(
                                val / diviseur, datetime.timezone.utc
                            )
                            dt_local = dt_utc.astimezone()
                            heure_str = f"{dt_local.hour:02d}:{dt_local.minute:02d}"
                            break
                    except Exception:
                        pass

            url_partants = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}/{num_r}/{num_c}/participants"
            try:
                # La stratégie de retry configurée plus haut s'applique aussi ici automatiquement
                res_part = session.get(url_partants, timeout=10)
                chevaux = []
                if res_part.status_code == 200:
                    for p in res_part.json().get("participants", []):
                        rapport_direct = p.get("dernierRapportDirect")
                        cote_val = (
                            rapport_direct.get("rapport")
                            if isinstance(rapport_direct, dict)
                            else None
                        )

                        rapport_ref = p.get("rapportReference")
                        cote_ouv = (
                            rapport_ref.get("rapport")
                            if isinstance(rapport_ref, dict)
                            else cote_val
                        )

                        tendance = "stable"
                        if cote_val and cote_ouv:
                            if cote_val < cote_ouv * 0.85:
                                tendance = "baisse_forte"
                            elif cote_val > cote_ouv * 1.15:
                                tendance = "hausse"

                        chevaux.append({
                            "num": p.get("numPmu"),
                            "nom": p.get("nom"),
                            "driver": p.get("driver", p.get("jockey", "")),
                            "musique": p.get("musique", ""),
                            "deferre": p.get("deferre", ""),
                            "poids": safe_float(p.get("poids", 0.0)),
                            "cote": cote_val,
                            "tendance_cote": tendance,
                        })
                resultats_journee.append({
                    "reunion": num_r,
                    "hippodrome": hippodrome,
                    "course": num_c,
                    "nom_course": nom_course,
                    "discipline": discipline,
                    "distance": distance_val,
                    "terrain_officiel": terrain_detecte,
                    "corde": corde_detectee,
                    "heure": heure_str,
                    "chevaux": chevaux,
                })
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des partants pour {num_r} {num_c} : {e}")

    if afficher_progres and progress_bar:
        progress_bar.empty()
        
    sauvegarder_courses_jour_db(date_iso, resultats_journee)
    
    # AVERTISSEMENT : st.cache_data.clear() vide l'intégralité du cache Streamlit.
    # Si d'autres éléments de l'app utilisent le cache, remplacez cette ligne par :
    # nom_de_la_fonction_a_vider.clear()
    st.cache_data.clear()
    
    logger.info(f"Téléchargement et mise en cache réussis pour la date {date_iso}.")
    return True