import datetime
import logging
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from utils.database import DB_PATH, sauvegarder_courses_jour_db
from utils.helpers import safe_float

logger = logging.getLogger("PMU_Pro")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

def creer_session_avec_retry():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

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

def _traiter_une_course(session, date_pmu, num_r, hippodrome, course):
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
    chevaux = []
    try:
        res_part = session.get(url_partants, timeout=10)
        if res_part.status_code == 200:
            for part_item in res_part.json().get("participants", []):
                rapport_direct = part_item.get("dernierRapportDirect")
                cote_val = (
                    rapport_direct.get("rapport")
                    if isinstance(rapport_direct, dict)
                    else None
                )

                rapport_ref = part_item.get("rapportReference")
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
                    "num": part_item.get("numPmu"),
                    "nom": part_item.get("nom"),
                    "driver": part_item.get("driver", part_item.get("jockey", "")),
                    "musique": part_item.get("musique", ""),
                    "deferre": part_item.get("deferre", ""),
                    "poids": safe_float(part_item.get("poids", 0.0)),
                    "cote": cote_val,
                    "tendance_cote": tendance,
                })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des partants pour {num_r} {num_c} : {e}")

    return {
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
    }

@st.cache_data
def telecharger_pmu_date(date_iso, fichier_cible=None, afficher_progres=True):
    try:
        dt = datetime.datetime.strptime(date_iso, "%Y-%m-%d")
        date_pmu = dt.strftime("%d%m%Y")
    except Exception as e:
        logger.warning(f"Format de date invalide pour l'API PMU ({date_iso}) : {e}")
        return False

    url_programme = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}"
    session = creer_session_avec_retry()

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

    taches_courses = []
    for reunion in reunions:
        num_r = f"R{reunion.get('numOfficiel')}"
        hippodrome = reunion.get("hippodrome", {}).get("libelleLong", "")
        for course in reunion.get("courses", []):
            taches_courses.append((session, date_pmu, num_r, hippodrome, course))

    total_courses = len(taches_courses)
    resultats_journee = []
    
    progress_bar = st.progress(0, text="Téléchargement des programmes en cours (Multi-thread)...") if afficher_progres else None
    courses_terminees = 0

    # Réduit à 5 threads max pour protéger le pool de connexions Supabase
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_traiter_une_course, *tache): tache for tache in taches_courses}
        
        for future in as_completed(futures):
            courses_terminees += 1
            if afficher_progres and progress_bar and total_courses > 0:
                prog_val = min(1.0, courses_terminees / total_courses)
                progress_bar.progress(prog_val, text=f"Téléchargement parallèle : {courses_terminees}/{total_courses} courses")
            try:
                res_course = future.result()
                if res_course:
                    resultats_journee.append(res_course)
            except Exception as e:
                logger.error(f"Erreur thread course : {e}")

    if afficher_progres and progress_bar:
        progress_bar.empty()
        
    if resultats_journee:
        sauvegarder_courses_jour_db(date_iso, resultats_journee)
        logger.info(f"Téléchargement et mise en cache réussis pour la date {date_iso}.")
        return True
    return False