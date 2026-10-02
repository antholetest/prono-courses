# utils/ai_model.py
import json
import logging
import math
import random
import re
import sqlite3
import numpy as np
import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from utils.database import DB_PATH, charger_historique, charger_courses_jour_db

logger = logging.getLogger("PMU_Pro")

def safe_float(val, default=0.0):
    if val is None or val == "" or val == "-":
        return default
    try:
        return float(str(val).replace(",", ".").replace("€", "").strip())
    except (ValueError, TypeError):
        return default


# --- MODULE IA & FEATURE ENGINEERING AVANCÉ (LIGHTGBM LEARNING TO RANK) ---
MODELE_IA_DEFAUT = {
    "poids_musique": 1.05,
    "poids_ferrage": 1.25,
    "poids_terrain": 1.15,
    "poids_poids": 1.0,
    "poids_cote_tendance": 1.35,
    "poids_driver": 1.15,
    "poids_corde": 1.0,
    "poids_hippodrome_acteur": 1.25,
    "poids_distance": 1.1,
    "poids_outsider_cache": 1.30,
    "seuil_value_bet": 1.50,
    "frequence_kelly": 0.05,
    "stats_impact": {
        "victoires_par_ferrage": 0,
        "victoires_par_smart_money": 0,
        "victoires_par_terrain": 0,
        "victoires_par_hippodrome": 0,
        "victoires_par_distance": 0,
        "total_analyses": 0,
        "gain_cumule_ia": 0.0,
    },
    "historique_ajustements": [],
}

def analyser_forme_musique_avancee(musique_str):
    """Analyse finement les 5 dernières courses avec pondération temporelle."""
    if not musique_str:
        return 0.0, 0, 0, 0
    
    m_clean = str(musique_str).upper().strip()
    tokens = re.findall(r'[1-90DATA]', m_clean)[:5]
    
    score_forme = 0.0
    victoires = 0
    podiums = 0
    fautes = 0
    
    poids_temporel = [1.5, 1.3, 1.1, 1.0, 0.9]
    
    for idx, token in enumerate(tokens):
        poids = poids_temporel[idx] if idx < len(poids_temporel) else 0.8
        if token == '1':
            score_forme += 10.0 * poids
            victoires += 1
            podiums += 1
        elif token in ['2', '3']:
            score_forme += 6.0 * poids
            podiums += 1
        elif token in ['4', '5']:
            score_forme += 2.0 * poids
        elif token in ['D', 'T', 'A', '0']:
            score_forme -= 4.0 * poids
            if token in ['D', 'T', 'A']:
                fautes += 1
                
    return round(score_forme, 2), victoires, podiums, fautes

def calculer_taux_driver_hippodrome_db(nom_driver, hippodrome_cible):
    """Calcule le taux de réussite (victoires/courses) d'un driver sur un hippodrome donné depuis la base SQLite."""
    if not nom_driver or not hippodrome_cible:
        return 0.0
    
    driver_upper = str(nom_driver).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    
    total_courses = 0
    total_victoires = 0
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM courses_cache")
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            race = json.loads(row[0])
            h_race = str(race.get("hippodrome", "")).upper().strip()
            if hipp_upper in h_race or h_race in hipp_upper:
                for part in race.get("chevaux", []):
                    d_part = str(part.get("driver", "")).upper().strip()
                    if d_part == driver_upper:
                        total_courses += 1
                        if part.get("ordreArrivee") == 1:
                            total_victoires += 1
    except Exception as e:
        logger.error(f"Erreur calcul taux driver/hippodrome : {e}")
        
    if total_courses == 0:
        return 0.0
    return round((total_victoires / total_courses) * 100.0, 2)

def extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome=""):
    """Extrait des variables numériques avancées (features enrichies) pour le modèle LightGBM LTR."""
    musique = cheval.get("musique", "")
    score_forme, nb_victoires, nb_podiums, fautes = analyser_forme_musique_avancee(musique)
    
    driver = cheval.get("driver", "")
    taux_driver_hipp = calculer_taux_driver_hippodrome_db(driver, hippodrome)
    
    cote = safe_float(cheval.get("cote"), 10.0)
    poids = safe_float(cheval.get("poids"), 0.0)
    
    is_trot = 1 if "Trot" in str(discipline) else 0
    is_lourd = 1 if terrain in ["Collant", "Lourd"] else 0
    
    return [
        score_forme,
        nb_victoires,
        nb_podiums,
        fautes,
        taux_driver_hipp,
        cote,
        poids,
        is_trot,
        is_lourd
    ]

@st.cache_resource
def entrainer_modele_ml_depuis_db():
    """Entraîne un modèle LightGBM Ranking avec Time-Series Split et évaluation continue."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT date_iso, data_json FROM courses_cache ORDER BY date_iso ASC")
        rows = cursor.fetchall()
        conn.close()

        races_features = []
        races_labels = []
        races_groups = []

        for date_iso, data_json in rows:
            try:
                race = json.loads(data_json)
                discipline = race.get("discipline", "Galop Plat")
                terrain = race.get("terrain_officiel", "Bon (Standard)")
                hippodrome = race.get("hippodrome", "")
                chevaux = race.get("chevaux", [])
                
                race_feats, race_labs = [], []
                for part in chevaux:
                    cote = safe_float(part.get("cote"), 0.0)
                    if cote <= 1.0:
                        continue
                    
                    features = extraire_caracteristiques_cheval(part, discipline, terrain, hippodrome)
                    ordre = part.get("ordreArrivee", 0)
                    
                    if ordre == 1: relevance = 4
                    elif ordre in [2, 3]: relevance = 3
                    elif ordre in [4, 5]: relevance = 1
                    else: relevance = 0
                        
                    race_feats.append(features)
                    race_labs.append(relevance)
                
                if len(race_feats) >= 3:
                    races_features.append(race_feats)
                    races_labels.append(race_labs)
                    races_groups.append(len(race_feats))
            except Exception:
                continue

        if len(races_groups) < 10:
            return None

        # 1. Validation croisée temporelle avec suivi des scores de validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(races_groups)))

        for train_idx, val_idx in tscv.split(races_groups):
            train_X, train_y, train_groups = [], [], []
            val_X, val_y, val_groups = [], [], []
            
            for idx in train_idx:
                train_X.extend(races_features[idx])
                train_y.extend(races_labels[idx])
                train_groups.append(races_groups[idx])
                
            for idx in val_idx:
                val_X.extend(races_features[idx])
                val_y.extend(races_labels[idx])
                val_groups.append(races_groups[idx])
                
            if not train_groups or not val_groups:
                continue
                
            temp_ranker = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=60,
                learning_rate=0.04,
                random_state=42,
                verbose=-1
            )
            
            temp_ranker.fit(
                np.array(train_X), np.array(train_y), group=np.array(train_groups),
                eval_set=[(np.array(val_X), np.array(val_y))],
                eval_group=[np.array(val_groups)],
                eval_metric="ndcg"
            )

        # 2. Entraînement du modèle final sur TOUTES les données disponibles
        X_all, y_all, groups_all = [], [], []
        for f_list, l_list, g_val in zip(races_features, races_labels, races_groups):
            X_all.extend(f_list)
            y_all.extend(l_list)
            groups_all.append(g_val)

        final_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=60,
            learning_rate=0.04,
            random_state=42,
            verbose=-1
        )
        final_ranker.fit(np.array(X_all), np.array(y_all), group=np.array(groups_all))

        return final_ranker
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement LightGBM : {e}")
        return None

def optimiser_poids_ia_automatique():
    """Optimise par recherche aléatoire les poids heuristiques du modèle en s'appuyant sur l'historique SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM courses_cache")
        rows = cursor.fetchall()
        conn.close()

        courses_evaluees = []
        for row in rows:
            race = json.loads(row[0])
            chevaux = race.get("chevaux", [])
            if any(p.get("ordreArrivee", 0) > 0 for p in chevaux):
                courses_evaluees.append(race)

        if len(courses_evaluees) < 20:
            return False, "Pas assez de courses avec résultats enregistrés (minimum 20 requis)."

        modele_actuel = charger_modele_ia()
        
        cles_poids = [
            "poids_musique", "poids_ferrage", "poids_terrain", 
            "poids_cote_tendance", "poids_driver", "poids_outsider_cache"
        ]
        
        meilleur_score_perf = -1
        meilleurs_poids = {k: modele_actuel.get(k, 1.0) for k in cles_poids}

        def evaluer_combinaison(poids_test):
            score_total = 0
            nb_tests = 0
            for race in courses_evaluees[:50]:
                for part in race.get("chevaux", []):
                    ordre = part.get("ordreArrivee", 0)
                    if ordre == 0:
                        continue
                    
                    musique = str(part.get("musique", ""))
                    cote = safe_float(part.get("cote"), 10.0)
                    
                    s = 0.0
                    if "1" in musique[:3]:
                        s += 10.0 * poids_test["poids_musique"]
                    if "QUATRE" in str(part.get("deferre", "")).upper():
                        s += 8.0 * poids_test["poids_ferrage"]
                    if part.get("tendance_cote") == "baisse_forte":
                        s += 6.0 * poids_test["poids_cote_tendance"]
                    if 6.0 <= cote <= 20.0:
                        s += 5.0 * poids_test["poids_outsider_cache"]
                        
                    if ordre <= 3:
                        score_total += s * (4 - ordre)
                    nb_tests += 1
            return score_total / max(1, nb_tests)

        for _ in range(50):
            poids_essai = {
                k: round(random.uniform(0.5, 2.5), 2) for k in cles_poids
            }
            perf = evaluer_combinaison(poids_essai)
            if perf > meilleur_score_perf:
                meilleur_score_perf = perf
                meilleurs_poids = poids_essai

        for k, v in meilleurs_poids.items():
            modele_actuel[k] = v
            
        sauvegarder_modele_ia(modele_actuel)
        logger.info(f"Optimisation automatique des poids réussie (Score: {meilleur_score_perf:.2f})")
        return True, f"Optimisation réussie ! Meilleurs poids appliqués (Score d'adéquation : {meilleur_score_perf:.2f})"
        
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation automatique des poids : {e}")
        return False, f"Erreur : {str(e)}"

def charger_modele_ia():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT cle, valeur FROM modele_ia")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        sauvegarder_modele_ia(MODELE_IA_DEFAUT.copy())
        return MODELE_IA_DEFAUT.copy()
    
    modele = {}
    for cle, val_json in rows:
        try:
            modele[cle] = json.loads(val_json)
        except Exception:
            modele[cle] = val_json
            
    for k, v in MODELE_IA_DEFAUT.items():
        if k not in modele:
            modele[k] = v
    return modele

def sauvegarder_modele_ia(modele):
    cles_poids = [
        "poids_musique",
        "poids_ferrage",
        "poids_terrain",
        "poids_poids",
        "poids_cote_tendance",
        "poids_driver",
        "poids_corde",
        "poids_hippodrome_acteur",
        "poids_distance",
        "poids_outsider_cache",
    ]
    for cle in cles_poids:
        if cle in modele:
            modele[cle] = max(0.1, min(3.0, float(modele[cle])))
            
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for k, v in modele.items():
        cursor.execute("""
            INSERT OR REPLACE INTO modele_ia (cle, valeur) VALUES (?, ?)
        """, (k, json.dumps(v)))
    conn.commit()
    conn.close()
    st.toast("Modèle IA enregistré localement !", icon="💾")

# --- PROTECTION PAR MOT DE PASSE ---
def verifier_authentification():
    if "authentifie" not in st.session_state:
        st.session_state["authentifie"] = False

    if not st.session_state["authentifie"]:
        st.title("🔒 Espace Restreint - Connexion Sécurisée")
        mot_de_passe_saisi = st.text_input(
            "Entrez votre mot de passe", type="password"
        )
        if st.button("Se connecter"):
            mdp_attendu = st.secrets.get("PASSWORD", "301180")
            if mot_de_passe_saisi.strip() == mdp_attendu:
                st.session_state["authentifie"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.stop()

verifier_authentification()

def reinitialiser_application_complete():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM paris")
    cursor.execute("DELETE FROM modele_ia")
    cursor.execute("DELETE FROM courses_cache")
    cursor.execute("DELETE FROM bilans_journee")
    conn.commit()
    conn.close()

    fichiers_supprimes = 0
    patterns = [
        "historique_paris.json",
        "modele_ia_pmu.json",
        "pmu_du_jour_*.json",
        "bilan_journee_*.json",
        "pmu_database.db"
    ]
    for pattern in patterns:
        for f in DOSSIER.glob(pattern):
            try:
                f.unlink()
                fichiers_supprimes += 1
            except Exception:
                pass

    for key in list(st.session_state.keys()):
        del st.session_state[key]
    logger.warning("Réinitialisation complète de l'application effectuée.")
    return fichiers_supprimes

# --- FONCTIONS DISCIPLINE & ENVIRONNEMENT ---
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

def telecharger_pmu_date(date_iso, fichier_cible):
    try:
        dt = datetime.datetime.strptime(date_iso, "%Y-%m-%d")
        date_pmu = dt.strftime("%d%m%Y")
    except Exception as e:
        logger.warning(f"Format de date invalide pour l'API PMU ({date_iso}) : {e}")
        return False

    url_programme = (
        f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}"
    )
    try:
        res = requests.get(url_programme, headers=HEADERS, timeout=15)
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
    
    progress_bar = st.progress(0, text="Téléchargement des programmes en cours...")

    for reunion in reunions:
        num_r = f"R{reunion.get('numOfficiel')}"
        hippodrome = reunion.get("hippodrome", {}).get("libelleLong", "")

        for course in reunion.get("courses", []):
            courses_traitees += 1
            if total_courses > 0:
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
                res_part = requests.get(url_partants, headers=HEADERS, timeout=10)
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

    progress_bar.empty()
    sauvegarder_courses_jour_db(date_iso, resultats_journee)
    st.cache_data.clear()
    logger.info(f"Téléchargement et mise en cache réussis pour la date {date_iso}.")
    return True

def analyser_affinite_distance(cheval, distance_course):
    if not distance_course:
        return 1.0

    dist_val = 0
    if isinstance(distance_course, (int, float)):
        dist_val = float(distance_course)
    else:
        m = re.search(r"(\d+)", str(distance_course))
        if m:
            dist_val = float(m.group(1))

    musique = str(cheval.get("musique") or "").upper()
    multiplicateur = 1.0

    if dist_val > 0:
        if "1" in musique[:4]:
            multiplicateur = 1.25
        elif "2" in musique[:4] or "3" in musique[:4]:
            multiplicateur = 1.12
        elif "0" in musique[:3] or "D" in musique[:3]:
            multiplicateur = 0.90
    else:
        multiplicateur = 1.05

    return round(multiplicateur, 2)

@st.cache_data(ttl=3600)
def analyser_performances_acteur_par_hippodrome(
    nom_acteur, hippodrome_cible
):
    if not nom_acteur:
        return 1.0
    acteur_upper = nom_acteur.upper().strip()
    hippodrome_upper = str(hippodrome_cible).upper().strip()

    apparitions_globales = 0
    apparitions_hippodrome = 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM courses_cache")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            try:
                race = json.loads(row[0])
                hipp_race = str(race.get("hippodrome", "")).upper().strip()
                est_meme_hippodrome = (
                    hippodrome_upper in hipp_race or hipp_race in hippodrome_upper
                )

                for part in race.get("chevaux", []):
                    driver_part = str(part.get("driver", "")).upper().strip()
                    if driver_part == acteur_upper:
                        apparitions_globales += 1
                        if est_meme_hippodrome:
                            apparitions_hippodrome += 1
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Erreur analyse performances acteur par hippodrome : {e}")

    bonus_hippodrome = min(apparitions_hippodrome * 1.0, 6.0)
    bonus_global = min(apparitions_globales * 0.2, 3.0)
    multiplicateur = 1.0 + ((bonus_hippodrome + bonus_global) / 10.0)
    return multiplicateur

def verifier_stop_loss(date_jour):
    historique = charger_historique()
    if not historique:
        return False
    try:
        perte = 0.0
        for p in historique:
            if str(p.get("date")) == str(date_jour) and p.get("statut") == "Perdu":
                m = safe_float(p.get("mise", 0))
                g = safe_float(p.get("gain", 0))
                if m > g:
                    perte += m - g

        return perte >= 35.0
    except Exception as e:
        logger.error(f"Erreur lecture stop-loss : {e}")
        return False

def ajuster_seuil_value_bet_dynamique():
    """Analyse les 30 derniers paris terminés et ajuste dynamiquement le seuil de Value Bet."""
    historique = charger_historique()
    paris_regles = [p for p in historique if p.get("statut") in ["Gagné", "Perdu"]]
    
    if len(paris_regles) < 15:
        return
        
    derniers_paris = paris_regles[-30:]
    mises = sum(safe_float(p.get("mise", 0)) for p in derniers_paris)
    gains = sum(safe_float(p.get("gain", 0)) for p in derniers_paris if p.get("statut") == "Gagné")
    
    if mises <= 0:
        return
        
    roi_recent = ((gains - mises) / mises) * 100
    modele_ia = charger_modele_ia()
    seuil_actuel = modele_ia.get("seuil_value_bet", 1.50)
    
    nouveau_seuil = seuil_actuel
    
    if roi_recent < -15.0:
        nouveau_seuil = min(2.00, seuil_actuel + 0.05)
    elif roi_recent > 20.0:
        nouveau_seuil = max(1.30, seuil_actuel - 0.05)
        
    if nouveau_seuil != seuil_actuel:
        modele_ia["seuil_value_bet"] = round(nouveau_seuil, 2)
        sauvegarder_modele_ia(modele_ia)
        logger.info(f"Seuil Value Bet ajusté dynamiquement à {nouveau_seuil} (ROI récent: {roi_recent:.1f}%)")

def normaliser_scores_chevaux(chevaux, cle_score="score_analyse"):
    """Ramène les scores calculés d'une course sur une échelle relative de 0 à 100."""
    if not chevaux:
        return chevaux
    score_max = max(
        (safe_float(c.get(cle_score, 0)) for c in chevaux), default=0.0
    )
    if score_max > 0:
        for c in chevaux:
            c[cle_score] = round(
                (safe_float(c.get(cle_score, 0)) / score_max) * 100, 1
            )
    return chevaux

def evaluer_score_cheval(
    cheval,
    discipline,
    terrain,
    corde,
    date_jour,
    params_adaptatifs,
    hippodrome="",
    distance_course="",
):
    modele_ia = charger_modele_ia()
    score = 0.0
    musique = str(cheval.get("musique") or "").upper()
    deferre = str(cheval.get("deferre") or "").upper()
    driver = str(cheval.get("driver") or "").upper()
    cote = safe_float(cheval.get("cote"), 0.0)
    poids = safe_float(cheval.get("poids", 0.0))
    tendance = cheval.get("tendance_cote", "stable")
    bonus_place = params_adaptatifs.get("bonus_place", 0)

    score_musique = 0
    for idx, char in enumerate(musique[:8]):
        if char == "1":
            score_musique += 12 if idx >= 3 else 14
        elif char == "2":
            score_musique += 8 + bonus_place
        elif char == "3":
            score_musique += 6 + bonus_place
        elif char in ["4", "5"]:
            score_musique += 2
        elif char in ["0", "D", "T", "A"]:
            score_musique -= 7 if (char in ["D", "T", "A"] and idx < 3) else 4
    score += score_musique * modele_ia.get("poids_musique", 1.05)

    if "Trot" in str(discipline):
        if "QUATRE" in deferre:
            score += 10.0 * modele_ia.get("poids_ferrage", 1.25)
        elif "ANTERIEURS" in deferre or "POSTERIEURS" in deferre:
            score += 5.5 * modele_ia.get("poids_ferrage", 1.25)
    else:
        if poids > 0:
            if poids < 55.0:
                score += 4.5 * modele_ia.get("poids_poids", 1.0)
            elif poids > 62.0:
                score -= 3.5 * modele_ia.get("poids_poids", 1.0)
        if terrain in ["Collant", "Lourd"] and (
            "LOURD" in musique or "SOUPLE" in musique
        ):
            score += 7.0 * modele_ia.get("poids_terrain", 1.15)

    poids_corde = modele_ia.get("poids_corde", 1.0)
    corde_str = str(corde).upper()
    if "GAUCHE" in corde_str and ("G" in musique or "GAUCHE" in musique):
        score += 4.5 * poids_corde
    elif "DROITE" in corde_str and ("D" in musique or "DROITE" in musique):
        score += 4.5 * poids_corde
    else:
        score += 1.0 * poids_corde

    if tendance == "baisse_forte":
        score += 6.5 * modele_ia.get("poids_cote_tendance", 1.35)
    elif tendance == "hausse":
        score -= 3.0 * modele_ia.get("poids_cote_tendance", 1.35)

    mult_acteur = analyser_performances_acteur_par_hippodrome(driver, hippodrome)
    bonus_acteur = (mult_acteur - 1.0) * 10.0
    score += (
        bonus_acteur
        * modele_ia.get("poids_driver", 1.15)
        * modele_ia.get("poids_hippodrome_acteur", 1.25)
    )

    poids_dist_ia = modele_ia.get("poids_distance", 1.1)
    mult_distance = analyser_affinite_distance(cheval, distance_course)
    bonus_distance = (mult_distance - 1.0) * 5.0
    score += bonus_distance * poids_dist_ia

    if cote > 1.0:
        if cote < 2.5:
            score += 6
        elif 2.5 <= cote <= 6.0:
            score += 8
        elif 6.0 < cote <= 18.0:
            score += 10
        elif cote > 35.0:
            score -= 2

    poids_outsider = modele_ia.get("poids_outsider_cache", 1.30)
    if 6.0 <= cote <= 30.0:
        bonus_joker = 0.0
        if tendance == "baisse_forte":
            bonus_joker += 9.0
        if "Trot" in str(discipline) and "QUATRE" in deferre:
            bonus_joker += 7.0
        if mult_acteur > 1.2:
            bonus_joker += 6.0

        impact_joker = min(15.0, bonus_joker * poids_outsider)
        score += impact_joker

    score += params_adaptatifs.get("malus_discipline", {}).get(discipline, 0)
    score_heuristique = max(0.0, round(score, 1))

    try:
        ml_model = entrainer_modele_ml_depuis_db()
        if ml_model is not None:
            features = np.array([extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome)])
            score_pred = ml_model.predict(features)[0]
            score_ml_normalise = max(0.0, min(100.0, (score_pred / 3.0) * 100.0))
            score_final = (score_heuristique * 0.3) + (score_ml_normalise * 0.7)
            return max(0.0, round(score_final, 1))
    except Exception as e:
        logger.debug(f"Modèle ML LightGBM non appliqué sur ce cheval : {e}")

    return score_heuristique

def calculer_valeur_esperee_avancee(chevaux_valides, nb_partants=12):
    if not chevaux_valides:
        return chevaux_valides

    temperature = 12.0
    scores = [safe_float(c.get("score_analyse", 0)) for c in chevaux_valides]
    max_score = max(scores, default=0.0)

    exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
    somme_exp = sum(exp_scores)
    nb_places = 3 if nb_partants >= 8 else 2

    for i, c in enumerate(chevaux_valides):
        proba_estimee = exp_scores[i] / somme_exp if somme_exp > 0 else 0.0
        c["proba_estimee"] = round(proba_estimee, 4)
        cote = safe_float(c.get("cote"), 0.0)
        c["ev_index"] = round(proba_estimee * cote, 2) if cote > 1.0 else 0.0

        proba_place = min(
            0.95, proba_estimee * (nb_places * 2.1) + (1.0 / nb_partants * 0.6)
        )
        c["proba_place"] = round(proba_place, 4)

        cote_place = max(1.1, 1.0 + (cote - 1.0) / (nb_places + 1.2))
        c["cote_place_estimee"] = round(cote_place, 2)
        c["ev_place_index"] = (
            round(proba_place * cote_place, 2) if cote > 1.0 else 0.0
        )
    return chevaux_valides

def calculer_fraction_kelly_exacte(p, c, frequence_kelly=0.05):
    if c <= 1.0 or p <= 0:
        return 0.0
    kelly = (p * c - 1.0) / (c - 1.0)
    if kelly <= 0:
        return 0.0
    return max(0.01, kelly * frequence_kelly)

def generer_plan_budget_journalier(date_iso, budget_base, params_adaptatifs):
    donnees, _ = charger_courses_jour_db(date_iso)
    budget_total_effectif = safe_float(budget_base)
    opportunites = []
    malus_disc = params_adaptatifs.get("malus_discipline", {})
    modele_ia = charger_modele_ia()
    seuil_min_ev = modele_ia.get("seuil_value_bet", 1.50)
    frequence_k = modele_ia.get("frequence_kelly", 0.05)

    for course in donnees:
        chevaux = course.get("chevaux", [])
        discipline = course.get("discipline", "Galop Plat")
        if malus_disc.get(discipline, 0) <= -3:
            continue

        terrain = course.get("terrain_officiel", "Bon (Standard)")
        corde = course.get("corde", "Corde standard")
        distance_course = course.get("distance", "")
        chevaux_valides = [
            c for c in chevaux
            if safe_float(c.get("cote")) > 1.0 or c.get("cote") is None
        ]
        nb_partants_total = len(chevaux)

        if len(chevaux_valides) < 3:
            continue

        for c in chevaux_valides:
            c["score_analyse"] = evaluer_score_cheval(
                c,
                discipline,
                terrain,
                corde,
                date_iso,
                params_adaptatifs,
                distance_course=distance_course,
            )

        normaliser_scores_chevaux(chevaux_valides, "score_analyse")
        calculer_valeur_esperee_avancee(chevaux_valides, nb_partants_total)

        chevaux_tries_score = sorted(
            chevaux_valides, key=lambda x: x["score_analyse"], reverse=True
        )
        chevaux_tries_ev = sorted(
            chevaux_valides, key=lambda x: x["ev_index"], reverse=True
        )

        meilleur_score = chevaux_tries_score[0]
        meilleur_ev = chevaux_tries_ev[0]

        if meilleur_ev["ev_index"] < seuil_min_ev:
            continue

        ecart_score = (
            meilleur_score["score_analyse"]
            - chevaux_tries_score[1]["score_analyse"]
            if len(chevaux_tries_score) > 1
            else 10.0
        )
        indice_confiance = ecart_score + (meilleur_ev["ev_index"] * 10)

        outsiders = [
            c
            for c in chevaux_valides
            if 7.0 <= safe_float(c.get("cote")) <= 22.0
            and c["num"] != meilleur_score["num"]
        ]
        poker = (
            max(outsiders, key=lambda x: x["ev_index"])
            if outsiders
            else (
                chevaux_tries_score[1]
                if len(chevaux_tries_score) > 1
                else meilleur_score
            )
        )

        r_nom_complet = (
            f"{course.get('reunion', 'R1')} -"
            f" {course.get('hippodrome', 'HIPPODROME')}"
        )

        p_place = safe_float(meilleur_score.get("proba_place", 0.3))
        c_place = safe_float(meilleur_score.get("cote_place_estimee", 2.0))
        kelly_secu = calculer_fraction_kelly_exacte(p_place, c_place, frequence_k)

        p_gagnant = safe_float(poker.get("proba_estimee", 0.1))
        c_gagnant = safe_float(poker.get("cote", 5.0))
        kelly_poker = calculer_fraction_kelly_exacte(
            p_gagnant, c_gagnant, frequence_k
        )

        opportunites.append({
            "score_confiance": max(1.0, indice_confiance),
            "ev_max": meilleur_ev["ev_index"],
            "kelly_combined": kelly_secu + kelly_poker,
            "reunion_course": f"{r_nom_complet} - {course.get('course')}",
            "reunion_clean": r_nom_complet,
            "nom_course": course.get("nom_course"),
            "discipline": discipline,
            "meilleur_cheval": meilleur_score,
            "poker": poker,
            "nb_partants": nb_partants_total,
        })

    opportunites.sort(
        key=lambda x: (x["kelly_combined"], x["ev_max"]), reverse=True
    )
    if not opportunites:
        return []

    max_courses = (
        1
        if budget_total_effectif < 25.0
        else (2 if budget_total_effectif < 60.0 else 3)
    )
    top_courses = opportunites[:max_courses]

    somme_kelly = sum(c["kelly_combined"] for c in top_courses)
    brutes_mises = (
        [
            (budget_total_effectif * (c["kelly_combined"] / somme_kelly))
            for c in top_courses
        ]
        if somme_kelly > 0
        else [budget_total_effectif / len(top_courses)] * len(top_courses)
    )
    mises_allouees = [max(1, int(round(m))) for m in brutes_mises]

    diff = int(budget_total_effectif) - sum(mises_allouees)
    if diff != 0 and mises_allouees:
        mises_allouees[0] = max(1, mises_allouees[0] + diff)

    plan_paris = []
    for idx, course_opt in enumerate(top_courses):
        mise_course = mises_allouees[idx]
        chev_base, chev_poker = course_opt["meilleur_cheval"], course_opt["poker"]

        ratio_secu = 0.80
        mise_secu = max(1, int(round(mise_course * ratio_secu)))
        mise_poker = max(0, mise_course - mise_secu)
        cote_poker = safe_float(chev_poker.get("cote"), 5.0)

        plan_paris.append({
            "Reunion_Clean": course_opt["reunion_clean"],
            "Course": course_opt["reunion_course"],
            "Discipline": course_opt["discipline"],
            "Base Value Bet (Sécurité)": (
                f"Simple Placé ➔ N°{chev_base['num']} - {chev_base['nom']} (Cote"
                f" Placé: {chev_base.get('cote_place_estimee', 0):.1f} | EV Place:"
                f" {chev_base.get('ev_place_index', 0):.2f})"
            ),
            "Mise Sécu": f"{mise_secu} €",
            "Coup de Poker Value": (
                f"Simple Gagnant ➔ N°{chev_poker['num']} - {chev_poker['nom']}"
                f" (Cote: {cote_poker:.1f} | EV: {chev_poker.get('ev_index', 0):.2f})"
                if mise_poker > 0
                else "Aucun"
            ),
            "Mise Poker": f"{mise_poker} €",
            "Mise Totale Course": f"{mise_course} €",
        })

    return plan_paris

def calculer_parametres_adaptatifs():
    ajuster_seuil_value_bet_dynamique()
    
    params = {
        "bonus_place": 0,
        "malus_discipline": {},
        "types_privilegies": ["Simple", "Couplé"],
        "message_auto": "Algorithme de maximisation des gains actif.",
    }
    historique = charger_historique()
    if not historique:
        return params
    try:
        paris_regles = [
            p for p in historique if p.get("statut") in ["Gagné", "Perdu"]
        ]
        derniers = paris_regles[-100:]
        if not derniers:
            return params

        perdus = [p for p in derniers if p.get("statut") == "Perdu"]
        messages = []
        if perdus:
            proche = sum(
                1 for p in perdus if "Quasi-podium" in str(p.get("diagnostic", ""))
            )
            taux = proche / len(perdus)
            if taux >= 0.2:
                params["bonus_place"] = int(round(taux * 12))
                messages.append(
                    f"🎯 {proche} quasi-podium(s) -> Bonus régularité (+{params['bonus_place']}"
                    " pts)"
                )

        roi_disc = {}
        for p in paris_regles:
            disc = p.get("discipline", "Galop Plat")
            if disc not in roi_disc:
                roi_disc[disc] = {"mises": 0.0, "gains": 0.0, "nb": 0}
            roi_disc[disc]["mises"] += safe_float(p.get("mise", 0))
            roi_disc[disc]["nb"] += 1
            if p.get("statut") == "Gagné":
                roi_disc[disc]["gains"] += safe_float(p.get("gain", 0))

        for disc, vals in roi_disc.items():
            if vals["nb"] >= 8 and vals["mises"] >= 50.0:
                mises_val = vals["mises"]
                if mises_val > 0:
                    roi = ((vals["gains"] - mises_val) / mises_val) * 100
                    if roi < -20.0:
                        params["malus_discipline"][disc] = -3
                        messages.append(
                            f"⚠️ Discipline '{disc}' en déficit ({roi:.1f}% ROI) -> Malus -3"
                            " pts"
                        )
                    elif roi > 10.0:
                        messages.append(
                            f"🔥 Discipline '{disc}' à haut ROI (+{roi:.1f}%) -> Priorisée"
                        )

        params["message_auto"] = (
            " | ".join(messages)
            if messages
            else "🤖 Modèle Value-Bet & Maximisation des Gains actif."
        )
    except Exception as e:
        logger.error(f"Erreur calcul paramètres adaptatifs : {e}")
    return params

def retroaction_apprentissage_ia(pari, arrivee_trouvee, cotes_reelles, liste_partants_bruts):
    """
    Analyse le résultat d'un pari terminé, met à jour les stats d'impact du modèle 
    et enregistre les gains/pertes cumulés.
    """
    statut = pari.get("statut", "Inconnu")
    mise = safe_float(pari.get("mise", 0))
    gain = safe_float(pari.get("gain", 0))
    profit = gain - mise

    modele = charger_modele_ia()
    stats = modele.get("stats_impact", {})
    
    # Incrémentation des statistiques d'impact
    stats["total_analyses"] = stats.get("total_analyses", 0) + 1
    stats["gain_cumule_ia"] = stats.get("gain_cumule_ia", 0.0) + profit

    details = str(pari.get("details", ""))
    if statut == "Gagné":
        if "déferré" in details.lower() or "quatuor" in details.lower() or "sécurité" in details.lower():
            stats["victoires_par_ferrage"] = stats.get("victoires_par_ferrage", 0) + 1

    modele["stats_impact"] = stats
    sauvegarder_modele_ia(modele)

    top_3 = ", ".join(arrivee_trouvee[:3]) if len(arrivee_trouvee) >= 3 else ", ".join(arrivee_trouvee)
    
    if statut == "Gagné":
        diagnostic = f"✅ Succès validé. Arrivée : [{top_3}]. Gain net : +{profit:.2f}€. Modèle mis à jour."
    else:
        diagnostic = f"❌ Échec enregistré. Arrivée réelle : [{top_3}]. Perte : {mise:.2f}€."
        
    return diagnostic
