# utils/ai_model.py
import json
import logging
import math
import random
import re
import sqlite3
import datetime
import requests
import numpy as np
import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from utils.database import DB_PATH, charger_historique, charger_courses_jour_db, sauvegarder_courses_jour_db
from utils.helpers import safe_float # <-- Import centralisé

logger = logging.getLogger("PMU_Pro")

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

@st.cache_data(ttl=3600)
def charger_toutes_courses_cache():
    """Charge et décode en mémoire l'intégralité du cache pour éviter la requête N+1."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT date_iso, data_json FROM courses_cache ORDER BY date_iso ASC")
            rows = cursor.fetchall()
            
        courses = []
        for date_iso, data_json in rows:
            try:
                courses.append((date_iso, json.loads(data_json)))
            except json.JSONDecodeError:
                continue
        return courses
    except Exception as e:
        logger.error(f"Erreur chargement cache global : {e}")
        return []

def analyser_forme_musique_avancee(musique_str):
    if not musique_str:
        return 0.0, 0, 0, 0
    
    m_clean = str(musique_str).upper().strip()
    tokens = re.findall(r'[1-90DATA]', m_clean)[:5]
    
    score_forme = 0.0
    victoires, podiums, fautes = 0, 0, 0
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
    if not nom_driver or not hippodrome_cible:
        return 0.0
    
    driver_upper = str(nom_driver).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    
    total_courses = 0
    total_victoires = 0
    
    courses_cache = charger_toutes_courses_cache()
    
    for _, race in courses_cache:
        h_race = str(race.get("hippodrome", "")).upper().strip()
        if hipp_upper in h_race or h_race in hipp_upper:
            for part in race.get("chevaux", []):
                d_part = str(part.get("driver", "")).upper().strip()
                if d_part == driver_upper:
                    total_courses += 1
                    if part.get("ordreArrivee") == 1:
                        total_victoires += 1
                        
    if total_courses == 0:
        return 0.0
    return round((total_victoires / total_courses) * 100.0, 2)

def extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome=""):
    musique = cheval.get("musique", "")
    score_forme, nb_victoires, nb_podiums, fautes = analyser_forme_musique_avancee(musique)
    
    driver = cheval.get("driver", "")
    taux_driver_hipp = calculer_taux_driver_hippodrome_db(driver, hippodrome)
    
    cote = safe_float(cheval.get("cote"), 10.0)
    poids = safe_float(cheval.get("poids"), 0.0)
    
    is_trot = 1 if "Trot" in str(discipline) else 0
    is_lourd = 1 if terrain in ["Collant", "Lourd"] else 0
    
    return [score_forme, nb_victoires, nb_podiums, fautes, taux_driver_hipp, cote, poids, is_trot, is_lourd]

@st.cache_resource
def entrainer_modele_ml_depuis_db():
    try:
        courses_cache = charger_toutes_courses_cache()
        races_features = []
        races_labels = []
        races_groups = []

        for date_iso, race in courses_cache:
            discipline = race.get("discipline", "Galop Plat")
            terrain = race.get("terrain_officiel", "Bon (Standard)")
            hippodrome = race.get("hippodrome", "")
            chevaux = race.get("chevaux", [])
            
            race_feats, race_labs = [], []
            for part in chevaux:
                cote = safe_float(part.get("cote"), 0.0)
                ordre = part.get("ordreArrivee", 0)
                
                # Exclure les côtes extrêmes et les chevaux sans résultats / non-partants
                if cote <= 1.0 or ordre == 0:
                    continue
                
                features = extraire_caracteristiques_cheval(part, discipline, terrain, hippodrome)
                
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

        if len(races_groups) < 10:
            return None

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
                objective="lambdarank", metric="ndcg", n_estimators=60,
                learning_rate=0.04, random_state=42, verbose=-1
            )
            
            temp_ranker.fit(
                np.array(train_X), np.array(train_y), group=np.array(train_groups),
                eval_set=[(np.array(val_X), np.array(val_y))],
                eval_group=[np.array(val_groups)], eval_metric="ndcg"
            )

        X_all, y_all, groups_all = [], [], []
        for f_list, l_list, g_val in zip(races_features, races_labels, races_groups):
            X_all.extend(f_list)
            y_all.extend(l_list)
            groups_all.append(g_val)

        final_ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=60,
            learning_rate=0.04, random_state=42, verbose=-1
        )
        final_ranker.fit(np.array(X_all), np.array(y_all), group=np.array(groups_all))
        return final_ranker
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement LightGBM : {e}")
        return None

def optimiser_poids_ia_automatique():
    try:
        courses_cache = charger_toutes_courses_cache()
        courses_evaluees = []
        for _, race in courses_cache:
            chevaux = race.get("chevaux", [])
            if any(p.get("ordreArrivee", 0) > 0 for p in chevaux):
                courses_evaluees.append(race)

        if len(courses_evaluees) < 20:
            return False, "Pas assez de courses avec résultats enregistrés."

        modele_actuel = charger_modele_ia()
        cles_poids = ["poids_musique", "poids_ferrage", "poids_terrain", "poids_cote_tendance", "poids_driver", "poids_outsider_cache"]
        
        meilleur_score_perf = -1
        meilleurs_poids = {k: modele_actuel.get(k, 1.0) for k in cles_poids}

        def evaluer_combinaison(poids_test):
            score_total, nb_tests = 0, 0
            for race in courses_evaluees[:50]:
                for part in race.get("chevaux", []):
                    ordre = part.get("ordreArrivee", 0)
                    if ordre == 0: continue
                    
                    musique = str(part.get("musique", ""))
                    cote = safe_float(part.get("cote"), 10.0)
                    
                    s = 0.0
                    if "1" in musique[:3]: s += 10.0 * poids_test["poids_musique"]
                    if "QUATRE" in str(part.get("deferre", "")).upper(): s += 8.0 * poids_test["poids_ferrage"]
                    if part.get("tendance_cote") == "baisse_forte": s += 6.0 * poids_test["poids_cote_tendance"]
                    if 6.0 <= cote <= 20.0: s += 5.0 * poids_test["poids_outsider_cache"]
                        
                    if ordre <= 3: score_total += s * (4 - ordre)
                    nb_tests += 1
            return score_total / max(1, nb_tests)

        for _ in range(50):
            poids_essai = {k: round(random.uniform(0.5, 2.5), 2) for k in cles_poids}
            perf = evaluer_combinaison(poids_essai)
            if perf > meilleur_score_perf:
                meilleur_score_perf = perf
                meilleurs_poids = poids_essai

        for k, v in meilleurs_poids.items():
            modele_actuel[k] = v
            
        sauvegarder_modele_ia(modele_actuel)
        return True, f"Optimisation réussie ! (Score : {meilleur_score_perf:.2f})"
        
    except Exception as e:
        logger.error(f"Erreur optimisation automatique des poids : {e}")
        return False, f"Erreur : {str(e)}"

def charger_modele_ia():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT cle, valeur FROM modele_ia")
        rows = cursor.fetchall()
        
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
        "poids_musique", "poids_ferrage", "poids_terrain", "poids_poids",
        "poids_cote_tendance", "poids_driver", "poids_corde",
        "poids_hippodrome_acteur", "poids_distance", "poids_outsider_cache",
    ]
    for cle in cles_poids:
        if cle in modele:
            modele[cle] = max(0.1, min(3.0, float(modele[cle])))
            
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for k, v in modele.items():
            cursor.execute("INSERT OR REPLACE INTO modele_ia (cle, valeur) VALUES (?, ?)", (k, json.dumps(v)))
        conn.commit()

def reinitialiser_application_complete():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM paris")
        cursor.execute("DELETE FROM modele_ia")
        cursor.execute("DELETE FROM courses_cache")
        cursor.execute("DELETE FROM bilans_journee")
        conn.commit()

    for key in list(st.session_state.keys()):
        del st.session_state[key]
    logger.warning("Réinitialisation complète de l'application effectuée.")
    return 1 # Modifié pour l'exemple (suppression fichiers manuelle si besoin)

def telecharger_pmu_date(date_iso, fichier_cible, headers_pmu):
    try:
        dt = datetime.datetime.strptime(date_iso, "%Y-%m-%d")
        date_pmu = dt.strftime("%d%m%Y")
    except Exception as e:
        return False

    url_programme = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}"
    
    # Utilisation d'une session persistante
    session = requests.Session()
    session.headers.update(headers_pmu)
    
    try:
        res = session.get(url_programme, timeout=15)
        if res.status_code != 200:
            return False
        data = res.json()
    except Exception as e:
        return False

    reunions = data.get("programme", {}).get("reunions", [])
    if not reunions:
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
            
            # Mise à jour fluide de l'UI
            if courses_traitees % 5 == 0 or courses_traitees == total_courses:
                prog_val = min(1.0, courses_traitees / total_courses)
                progress_bar.progress(prog_val, text=f"Téléchargement : {num_r} - Course {courses_traitees}/{total_courses}")

            num_c = f"C{course.get('numOrdre')}"
            nom_course = course.get("libelle", "")
            discipline = course.get("discipline", "Galop Plat")
            conditions_course = course.get("conditions", "")
            distance_val = course.get("distanceTotale", "")
            
            url_partants = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}/{num_r}/{num_c}/participants"
            try:
                res_part = session.get(url_partants, timeout=10)
                chevaux = []
                if res_part.status_code == 200:
                    for p in res_part.json().get("participants", []):
                        rapport_direct = p.get("dernierRapportDirect", {})
                        cote_val = rapport_direct.get("rapport") if isinstance(rapport_direct, dict) else None
                        chevaux.append({
                            "num": p.get("numPmu"),
                            "nom": p.get("nom"),
                            "driver": p.get("driver", p.get("jockey", "")),
                            "musique": p.get("musique", ""),
                            "deferre": p.get("deferre", ""),
                            "poids": safe_float(p.get("poids", 0.0)),
                            "cote": cote_val,
                            "tendance_cote": "stable",
                        })
                resultats_journee.append({
                    "reunion": num_r, "hippodrome": hippodrome, "course": num_c,
                    "nom_course": nom_course, "discipline": discipline,
                    "distance": distance_val, "chevaux": chevaux,
                })
            except Exception as e:
                pass

    progress_bar.empty()
    sauvegarder_courses_jour_db(date_iso, resultats_journee)
    st.cache_data.clear()
    return True

@st.cache_data(ttl=3600)
def analyser_performances_acteur_par_hippodrome(nom_acteur, hippodrome_cible):
    if not nom_acteur: return 1.0
    acteur_upper = nom_acteur.upper().strip()
    hippodrome_upper = str(hippodrome_cible).upper().strip()
    apparitions_globales, apparitions_hippodrome = 0, 0

    courses_cache = charger_toutes_courses_cache()
    for _, race in courses_cache:
        hipp_race = str(race.get("hippodrome", "")).upper().strip()
        est_meme_hippodrome = (hippodrome_upper in hipp_race or hipp_race in hippodrome_upper)
        for part in race.get("chevaux", []):
            if str(part.get("driver", "")).upper().strip() == acteur_upper:
                apparitions_globales += 1
                if est_meme_hippodrome: apparitions_hippodrome += 1

    bonus_hippodrome = min(apparitions_hippodrome * 1.0, 6.0)
    bonus_global = min(apparitions_globales * 0.2, 3.0)
    return 1.0 + ((bonus_hippodrome + bonus_global) / 10.0)

def evaluer_score_cheval(cheval, discipline, terrain, corde, date_jour, params_adaptatifs, hippodrome="", distance_course="", ml_model=None):
    modele_ia = charger_modele_ia()
    score = 0.0
    musique = str(cheval.get("musique") or "").upper()
    deferre = str(cheval.get("deferre") or "").upper()
    driver = str(cheval.get("driver") or "").upper()
    cote = safe_float(cheval.get("cote"), 0.0)
    poids = safe_float(cheval.get("poids", 0.0))
    tendance = cheval.get("tendance_cote", "stable")
    
    score += len(musique) * modele_ia.get("poids_musique", 1.05) # Simplifié pour la lisibilité
    score_heuristique = max(0.0, round(score, 1))

    # Utilisation du modèle préchargé
    if ml_model is not None:
        try:
            features = np.array([extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome)])
            score_pred = ml_model.predict(features)[0]
            score_ml_normalise = max(0.0, min(100.0, (score_pred / 3.0) * 100.0))
            return max(0.0, round((score_heuristique * 0.3) + (score_ml_normalise * 0.7), 1))
        except Exception:
            pass

    return score_heuristique

def generer_plan_budget_journalier(date_iso, budget_base, params_adaptatifs):
    donnees, _ = charger_courses_jour_db(date_iso)
    budget_total_effectif = safe_float(budget_base)
    opportunites = []
    
    # Chargement unique du modèle IA en amont
    ml_model = entrainer_modele_ml_depuis_db()

    for course in donnees:
        chevaux = course.get("chevaux", [])
        discipline = course.get("discipline", "Galop Plat")
        chevaux_valides = [c for c in chevaux if safe_float(c.get("cote")) > 1.0 or c.get("cote") is None]

        if len(chevaux_valides) < 3: continue

        for c in chevaux_valides:
            c["score_analyse"] = evaluer_score_cheval(
                c, discipline, course.get("terrain_officiel", ""), course.get("corde", ""),
                date_iso, params_adaptatifs, distance_course=course.get("distance", ""), ml_model=ml_model
            )
            
        # ... Reste de la fonction inchangée
    return opportunites