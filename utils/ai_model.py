# utils/ai_model.py
import json
import logging
import math
import random
import re
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from utils.database import DB_PATH, charger_historique, charger_courses_jour_db
from utils.helpers import safe_float

logger = logging.getLogger("PMU_Pro")

# --- CONFIGURATION IA ET HEURISTIQUES ---
MODELE_IA_DEFAUT = {
    "poids_musique": 1.05,
    "poids_ferrage": 1.25,
    "poids_terrain": 1.15,
    "poids_poids": 1.0,
    "poids_cote_tendance": 1.35,
    "poids_driver": 1.15,
    "poids_corde": 1.0,
    "poids_hippodrome_acteur": 1.25,
    "poids_distance": 1.10,
    "poids_outsider_cache": 1.30,
    "seuil_value_bet": 1.85,
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
    if not musique_str:
        return 0.0, 0, 0, 0
    m_clean = str(musique_str).upper().strip()
    tokens = re.findall(r"[1-90DATA]", m_clean)[:5]
    score_forme, victoires, podiums, fautes = 0.0, 0, 0, 0
    poids_temporel = [1.5, 1.3, 1.1, 1.0, 0.9]

    for idx, token in enumerate(tokens):
        poids = poids_temporel[idx] if idx < len(poids_temporel) else 0.8
        if token == "1":
            score_forme += 10.0 * poids
            victoires += 1
            podiums += 1
        elif token in ["2", "3"]:
            score_forme += 6.0 * poids
            podiums += 1
        elif token in ["4", "5"]:
            score_forme += 2.0 * poids
        elif token in ["D", "T", "A", "0"]:
            score_forme -= 4.0 * poids
            if token in ["D", "T", "A"]:
                fautes += 1

    return round(score_forme, 2), victoires, podiums, fautes

def calculer_taux_driver_hippodrome_db(nom_driver, hippodrome_cible):
    if not nom_driver or not hippodrome_cible:
        return 0.0
    driver_upper = str(nom_driver).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    total_courses, total_victoires = 0, 0

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
                    if str(part.get("driver", "")).upper().strip() == driver_upper:
                        total_courses += 1
                        if part.get("ordreArrivee") == 1:
                            total_victoires += 1
    except Exception as e:
        logger.error(f"Erreur calcul taux driver/hippodrome : {e}")

    return round((total_victoires / total_courses) * 100.0, 2) if total_courses > 0 else 0.0

def extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome=""):
    score_forme, nb_victoires, nb_podiums, fautes = analyser_forme_musique_avancee(cheval.get("musique", ""))
    taux_driver_hipp = calculer_taux_driver_hippodrome_db(cheval.get("driver", ""), hippodrome)
    cote = safe_float(cheval.get("cote"), 10.0)
    poids = safe_float(cheval.get("poids"), 0.0)
    deferre_str = str(cheval.get("deferre", "")).upper()
    is_deferre_4 = 1 if "QUATRE" in deferre_str else 0
    tendance = str(cheval.get("tendance_cote", "stable"))
    tendance_val = 1.0 if tendance == "baisse_forte" else (-1.0 if tendance == "hausse" else 0.0)
    is_trot = 1 if "Trot" in str(discipline) else 0
    is_lourd = 1 if terrain in ["Collant", "Lourd"] else 0

    return [
        score_forme, nb_victoires, nb_podiums, fautes,
        taux_driver_hipp, cote, poids, is_deferre_4,
        tendance_val, is_trot, is_lourd
    ]

def _entrainer_modele_ml_depuis_db_impl():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date_iso, data_json FROM courses_cache
            WHERE date_iso >= date('now', '-12 months') ORDER BY date_iso ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        races_features, races_labels, races_groups = [], [], []
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
                    relevance = 5 if ordre == 1 else 0
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
        logger.error(f"Erreur entraînement LightGBM : {e}")
        return None

@st.cache_resource
def charger_modele_ml_cached():
    return _entrainer_modele_ml_depuis_db_impl()

def rafraichir_modele_ml():
    try:
        charger_modele_ml_cached.clear()
    except Exception:
        pass
    return _entrainer_modele_ml_depuis_db_impl()

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
        "poids_musique", "poids_ferrage", "poids_terrain", "poids_poids",
        "poids_cote_tendance", "poids_driver", "poids_corde",
        "poids_hippodrome_acteur", "poids_distance", "poids_outsider_cache"
    ]
    for cle in cles_poids:
        if cle in modele:
            modele[cle] = max(0.1, min(3.0, float(modele[cle])))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for k, v in modele.items():
        cursor.execute("INSERT OR REPLACE INTO modele_ia (cle, valeur) VALUES (?, ?)", (k, json.dumps(v)))
    conn.commit()
    conn.close()

def optimiser_poids_ia_automatique():
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
            return False, "Pas assez de courses enregistrées (minimum 20 requises)."

        modele_actuel = charger_modele_ia()
        cles_poids = ["poids_musique", "poids_ferrage", "poids_terrain", "poids_cote_tendance", "poids_driver", "poids_outsider_cache"]
        meilleur_score = -1
        meilleurs_poids = {k: modele_actuel.get(k, 1.0) for k in cles_poids}

        def evaluer_combinaison(poids_test):
            score_total, nb_tests = 0, 0
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
            poids_essai = {k: round(random.uniform(0.5, 2.5), 2) for k in cles_poids}
            perf = evaluer_combinaison(poids_essai)
            if perf > meilleur_score:
                meilleur_score = perf
                meilleurs_poids = poids_essai

        for k, v in meilleurs_poids.items():
            modele_actuel[k] = v
        sauvegarder_modele_ia(modele_actuel)
        return True, f"Optimisation réussie ! Meilleurs poids appliqués (Score: {meilleur_score:.2f})"
    except Exception as e:
        return False, f"Erreur : {str(e)}"

def normaliser_scores_chevaux(chevaux, cle_score="score_analyse"):
    if not chevaux:
        return chevaux
    score_max = max((safe_float(c.get(cle_score, 0)) for c in chevaux), default=0.0)
    if score_max > 0:
        for c in chevaux:
            c[cle_score] = round((safe_float(c.get(cle_score, 0)) / score_max) * 100, 1)
    return chevaux

def analyser_affinite_distance(cheval, distance_course):
    if not distance_course:
        return 1.0
    dist_val = float(re.search(r"(\d+)", str(distance_course)).group(1)) if re.search(r"(\d+)", str(distance_course)) else 0.0
    musique = str(cheval.get("musique") or "").upper()
    if dist_val > 0:
        if "1" in musique[:4]:
            return 1.25
        elif "2" in musique[:4] or "3" in musique[:4]:
            return 1.12
        elif "0" in musique[:3] or "D" in musique[:3]:
            return 0.90
    return 1.05

@st.cache_data(ttl=3600)
def analyser_performances_acteur_par_hippodrome(nom_acteur, hippodrome_cible):
    if not nom_acteur:
        return 1.0
    acteur_upper = nom_acteur.upper().strip()
    hippodrome_upper = str(hippodrome_cible).upper().strip()
    apparitions_globales, apparitions_hippodrome = 0, 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM courses_cache")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            race = json.loads(row[0])
            hipp_race = str(race.get("hippodrome", "")).upper().strip()
            est_meme_hipp = hippodrome_upper in hipp_race or hipp_race in hippodrome_upper
            for part in race.get("chevaux", []):
                if str(part.get("driver", "")).upper().strip() == acteur_upper:
                    apparitions_globales += 1
                    if est_meme_hipp:
                        apparitions_hippodrome += 1
    except Exception as e:
        logger.error(f"Erreur analyse performances acteur : {e}")

    bonus_hipp = min(apparitions_hippodrome * 1.0, 6.0)
    bonus_glob = min(apparitions_globales * 0.2, 3.0)
    return 1.0 + ((bonus_hipp + bonus_glob) / 10.0)

def evaluer_score_cheval(cheval, discipline, terrain, corde, date_jour, params_adaptatifs, hippodrome="", distance_course=""):
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
        if terrain in ["Collant", "Lourd"] and ("LOURD" in musique or "SOUPLE" in musique):
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
    score += bonus_acteur * modele_ia.get("poids_driver", 1.15) * modele_ia.get("poids_hippodrome_acteur", 1.25)

    mult_distance = analyser_affinite_distance(cheval, distance_course)
    bonus_distance = (mult_distance - 1.0) * 5.0
    score += bonus_distance * modele_ia.get("poids_distance", 1.10)

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
        score += min(15.0, bonus_joker * poids_outsider)

    score += params_adaptatifs.get("malus_discipline", {}).get(discipline, 0)
    score_heuristique = max(0.0, round(score, 1))

    try:
        ml_model = charger_modele_ml_cached()
        if ml_model is not None:
            features = np.array([extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome)])
            score_pred = float(ml_model.predict(features)[0])
            score_ml_normalise = max(0.0, min(100.0, (score_pred / 3.0) * 100.0))
            score_final = (score_heuristique * 0.3) + (score_ml_normalise * 0.7)
            return max(0.0, round(score_final, 1))
    except Exception as e:
        logger.debug(f"Modèle ML LightGBM non appliqué : {e}")

    return score_heuristique

def calculer_valeur_esperee_avancee(chevaux_valides, nb_partants=12):
    if not chevaux_valides:
        return chevaux_valides
    temperature = 8.5
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

        proba_place = min(0.95, proba_estimee * (nb_places * 2.1) + (1.0 / nb_partants * 0.6))
        c["proba_place"] = round(proba_place, 4)
        cote_place = max(1.1, 1.0 + (cote - 1.0) / (nb_places + 1.2))
        c["cote_place_estimee"] = round(cote_place, 2)
        c["ev_place_index"] = round(proba_place * cote_place, 2) if cote > 1.0 else 0.0

    return chevaux_valides

def calculer_fraction_kelly_exacte(p, c, frequence_kelly=0.05):
    if c <= 1.0 or p <= 0:
        return 0.0
    kelly = (p * c - 1.0) / (c - 1.0)
    return max(0.01, kelly * frequence_kelly) if kelly > 0 else 0.0

def calculer_parametres_adaptatifs():
    historique = charger_historique()
    paris_regles = [p for p in historique if p.get("statut") in ["Gagné", "Perdu"]]
    params = {"bonus_place": 0, "malus_discipline": {}, "message_auto": "Modèle standard actif."}

    if len(paris_regles) < 10:
        return params

    mises = sum(safe_float(p.get("mise", 0)) for p in paris_regles)
    gains = sum(safe_float(p.get("gain", 0)) for p in paris_regles if p.get("statut") == "Gagné")
    roi_global = ((gains - mises) / mises * 100) if mises > 0 else 0.0

    if roi_global < -10.0:
        params["bonus_place"] = 2
        params["message_auto"] = f"ROI global faible ({roi_global:.1f}%) : renforcement du poids des places."

    return params

def generer_plan_budget_journalier(date_iso, budget_total=50.0, params_adaptatifs=None):
    donnees_jour, _ = charger_courses_jour_db(date_iso)
    if not donnees_jour:
        return []

    modele_ia = charger_modele_ia()
    seuil_ev = modele_ia.get("seuil_value_bet", 1.85)
    frequence_k = modele_ia.get("frequence_kelly", 0.05)

    opportunites = []
    for race in donnees_jour:
        chevaux = race.get("chevaux", [])
        if not chevaux:
            continue
        for c in chevaux:
            c["score_analyse"] = evaluer_score_cheval(c, race.get("discipline"), race.get("terrain_officiel"), race.get("corde"), date_iso, params_adaptatifs or {})
        normaliser_scores_chevaux(chevaux, "score_analyse")
        calculer_valeur_esperee_avancee(chevaux, len(chevaux))

        for c in chevaux:
            ev_g = safe_float(c.get("ev_index"), 0.0)
            ev_p = safe_float(c.get("ev_place_index"), 0.0)
            if ev_g >= seuil_ev or ev_p >= seuil_ev:
                opportunites.append({
                    "Réunion": race.get("reunion"),
                    "Course": race.get("course"),
                    "Cheval": c.get("nom"),
                    "N°": c.get("num"),
                    "Type": "Simple Gagnant" if ev_g >= ev_p else "Simple Placé",
                    "Cote": safe_float(c.get("cote")),
                    "EV": max(ev_g, ev_p),
                    "Proba": c.get("proba_estimee") if ev_g >= ev_p else c.get("proba_place")
                })

    if not opportunites:
        return []

    df_opp = pd.DataFrame(opportunites)
    df_opp["kelly_f"] = df_opp.apply(lambda r: calculer_fraction_kelly_exacte(r["Proba"], r["Cote"], frequence_k), axis=1)
    tot_f = df_opp["kelly_f"].sum()

    if tot_f > 0:
        df_opp["Mise (€)"] = ((df_opp["kelly_f"] / tot_f) * budget_total).round(1)
    else:
        df_opp["Mise (€)"] = round(budget_total / len(df_opp), 1)

    return df_opp[["Réunion", "Course", "N°", "Cheval", "Type", "Cote", "EV", "Mise (€)"]]

def retroaction_apprentissage_ia(pari, arrivee_trouvee, cotes_reelles, partants_bruts):
    if not arrivee_trouvee:
        return "Pas de résultat disponible."
    
    modele_ia = charger_modele_ia()
    stats = modele_ia.get("stats_impact", {})
    stats["total_analyses"] = stats.get("total_analyses", 0) + 1

    gagnant = arrivee_trouvee[0]
    gain = safe_float(pari.get("gain", 0))
    mise = safe_float(pari.get("mise", 0))
    stats["gain_cumule_ia"] = round(stats.get("gain_cumule_ia", 0.0) + (gain - mise), 2)

    sauvegarder_modele_ia(modele_ia)
    
    # Ajustement automatique du seuil EV selon le ROI récent
    ajuster_seuil_value_bet_dynamique()
    
    return f"Résultat traité | Premier : N°{gagnant}"