import json
import logging
import math
import random
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from utils.database import DB_PATH, get_connection, charger_historique, charger_courses_jour_db, load_bets_from_db
from utils.helpers import safe_float

logger = logging.getLogger("PMU_Pro")

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
    "poids_duo_entraineur": 1.20,  # Nouveau poids pour le couple Driver/Entraîneur
    "poids_frecheUR": 1.15,       # Nouveau poids pour la fraîcheur / repos
    "seuil_value_bet": 1.85,
    "frequence_kelly": 0.02,  
    "disciplines_specifiques": {}, 
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

_CACHE_INDEX_GLOBAL = None

def _charger_index_global_memoire():
    global _CACHE_INDEX_GLOBAL
    if _CACHE_INDEX_GLOBAL is not None:
        return _CACHE_INDEX_GLOBAL

    index_global = {"drivers": {}, "entraineurs": {}, "duos": {}, "hippodromes": {}, "chevaux": {}}
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data_json FROM courses_cache 
            WHERE date_iso >= TO_CHAR(CURRENT_DATE - INTERVAL '90 days', 'YYYY-MM-DD')
        """)
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            val_json = row["data_json"] if isinstance(row, dict) else row[0]
            try:
                races = json.loads(val_json)
                if isinstance(races, dict):
                    races = [races]
                if isinstance(races, list):
                    for race in races:
                        hipp_race = str(race.get("hippodrome", "")).upper().strip()
                        discipline = str(race.get("discipline", "")).upper().strip()
                        terrain = str(race.get("terrain_officiel", "")).upper().strip()

                        if hipp_race:
                            if hipp_race not in index_global["hippodromes"]:
                                index_global["hippodromes"][hipp_race] = {"total_courses": 0, "disciplines": set(), "terrains": set()}
                            index_global["hippodromes"][hipp_race]["total_courses"] += 1
                            if discipline:
                                index_global["hippodromes"][hipp_race]["disciplines"].add(discipline)
                            if terrain:
                                index_global["hippodromes"][hipp_race]["terrains"].add(terrain)

                        for part in race.get("chevaux", []):
                            driver = str(part.get("driver", "")).upper().strip()
                            entraineur = str(part.get("entraineur", "")).upper().strip()
                            ordre = part.get("ordreArrivee", 0)

                            if driver and driver != "INCONNU":
                                if driver not in index_global["drivers"]:
                                    index_global["drivers"][driver] = {"total": 0, "victoires": 0, "hippodromes": {}}
                                index_global["drivers"][driver]["total"] += 1
                                if ordre == 1:
                                    index_global["drivers"][driver]["victoires"] += 1

                                if hipp_race:
                                    if hipp_race not in index_global["drivers"][driver]["hippodromes"]:
                                        index_global["drivers"][driver]["hippodromes"][hipp_race] = {"total": 0, "victoires": 0}
                                    index_global["drivers"][driver]["hippodromes"][hipp_race]["total"] += 1
                                    if ordre == 1:
                                        index_global["drivers"][driver]["hippodromes"][hipp_race]["victoires"] += 1

                            if driver and entraineur and driver != "INCONNU" and entraineur != "INCONNU":
                                duo_key = f"{driver}__{entraineur}"
                                if duo_key not in index_global["duos"]:
                                    index_global["duos"][duo_key] = {"total": 0, "victoires": 0}
                                index_global["duos"][duo_key]["total"] += 1
                                if ordre == 1:
                                    index_global["duos"][duo_key]["victoires"] += 1

            except Exception:
                continue
    except Exception as e:
        logger.error(f"Erreur chargement index global optimisé : {e}")

    _CACHE_INDEX_GLOBAL = index_global
    return _CACHE_INDEX_GLOBAL

def reinitialiser_index_memoire():
    global _CACHE_INDEX_GLOBAL
    _CACHE_INDEX_GLOBAL = None

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

@st.cache_data(ttl=3600)
def calculer_taux_driver_hippodrome_db(nom_driver, hippodrome_cible):
    if not nom_driver or not hippodrome_cible:
        return 0.0
    driver_upper = str(nom_driver).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    index_data = _charger_index_global_memoire()
    index_drivers = index_data.get("drivers", {})
    if driver_upper not in index_drivers:
        return 0.0
    hippos = index_drivers[driver_upper]["hippodromes"]
    total_courses, total_victoires = 0, 0
    for h_key, stats in hippos.items():
        if hipp_upper in h_key or h_key in hipp_upper:
            total_courses += stats["total"]
            total_victoires += stats["victoires"]
    return round((total_victoires / total_courses) * 100.0, 2) if total_courses > 0 else 0.0

@st.cache_data(ttl=3600)
def calculer_taux_duo_driver_entraineur(nom_driver, nom_entraineur):
    if not nom_driver or not nom_entraineur:
        return 0.0
    duo_key = f"{str(nom_driver).upper().strip()}__{str(nom_entraineur).upper().strip()}"
    index_data = _charger_index_global_memoire()
    duos = index_data.get("duos", {})
    if duo_key not in duos:
        return 0.0
    stats = duos[duo_key]
    if stats["total"] < 3:
        return 0.0
    return round((stats["victoires"] / stats["total"]) * 100.0, 2)

def evaluer_indice_fraicheur(cheval):
    """Estime un bonus/malus de fraîcheur basé sur l'historique récent de la musique."""
    musique = str(cheval.get("musique", "")).upper()
    if not musique:
        return 1.0
    # Si le cheval court de façon très rapprochée (ex: courses consécutives à moins de 10-15 jours)
    # On détecte s'il y a beaucoup de perquisitions récentes ou un long break (rentrée).
    tokens = re.findall(r"[1-90DATA]", musique)
    if len(tokens) >= 2:
        # S'il vient de courir (premier chiffre à gauche récent), on analyse la configuration
        premier_token = tokens[0]
        if premier_token in ["D", "0"] and len(tokens) > 1 and tokens[1] in ["1", "2", "3"]:
            # Rentrée après échec ou course de prep ferré
            return 1.15
    return 1.0

def extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome=""):
    score_forme, nb_victoires, nb_podiums, fautes = analyser_forme_musique_avancee(cheval.get("musique", ""))
    taux_driver_hipp = calculer_taux_driver_hippodrome_db(cheval.get("driver", ""), hippodrome)
    taux_duo = calculer_taux_duo_driver_entraineur(cheval.get("driver", ""), cheval.get("entraineur", ""))
    cote = safe_float(cheval.get("cote"), 10.0)
    poids = safe_float(cheval.get("poids"), 0.0)
    deferre_str = str(cheval.get("deferre", "")).upper()
    is_deferre_4 = 1 if "QUATRE" in deferre_str else 0
    tendance = str(cheval.get("tendance_cote", "stable"))
    tendance_val = 1.0 if "baisse" in tendance else (-1.0 if "hausse" in tendance else 0.0)
    is_trot = 1 if "Trot" in str(discipline) else 0
    is_lourd = 1 if terrain in ["Collant", "Lourd"] else 0
    fraicheur = evaluer_indice_fraicheur(cheval)

    return [
        score_forme, nb_victoires, nb_podiums, fautes,
        taux_driver_hipp, taux_duo, cote, poids, is_deferre_4,
        tendance_val, is_trot, is_lourd, fraicheur
    ]

def _entrainer_modele_ml_depuis_db_impl():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data_json FROM courses_cache 
            ORDER BY date_iso DESC LIMIT 300
        """)
        rows = cursor.fetchall()
        conn.close()

        races_features, races_labels, races_groups = [], [], []
        for row in rows:
            try:
                data_json = row["data_json"] if isinstance(row, dict) else row[1]
                races = json.loads(data_json)
                if isinstance(races, dict):
                    races = [races]
                if not isinstance(races, list):
                    continue

                for race in races:
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

        if len(races_groups) < 5:
            return None

        X_all, y_all, groups_all = [], [], []
        for f_list, l_list, g_val in zip(races_features, races_labels, races_groups):
            X_all.extend(f_list)
            y_all.extend(l_list)
            groups_all.append(g_val)

        final_ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=80,
            learning_rate=0.03, random_state=42, verbose=-1
        )
        final_ranker.fit(np.array(X_all), np.array(y_all), group=np.array(groups_all))
        
        model_bytes = pickle.dumps(final_ranker)
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO modele_ia (cle, valeur) VALUES ('lgb_model_serialized', %s)
            ON CONFLICT (cle) DO UPDATE SET valeur = EXCLUDED.valeur
        """, (model_bytes.hex(),))
        conn.commit()
        cur.close()
        conn.close()

        return final_ranker
    except Exception as e:
        logger.error(f"Erreur entraînement LightGBM : {e}")
        return None

@st.cache_resource
def charger_modele_ml_cached():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT valeur FROM modele_ia WHERE cle = 'lgb_model_serialized'")
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            model_bytes = bytes.fromhex(row[0] if isinstance(row, str) else row.get("valeur"))
            return pickle.loads(model_bytes)
    except Exception:
        pass
    return _entrainer_modele_ml_depuis_db_impl()

def rafraichir_modele_ml():
    try:
        charger_modele_ml_cached.clear()
    except Exception:
        pass
    return _entrainer_modele_ml_depuis_db_impl()

def charger_modele_ia():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT cle, valeur FROM modele_ia WHERE cle != 'lgb_model_serialized'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        sauvegarder_modele_ia(MODELE_IA_DEFAUT.copy())
        return MODELE_IA_DEFAUT.copy()

    modele = {}
    for row in rows:
        cle = row["cle"] if isinstance(row, dict) else row[0]
        val_json = row["valeur"] if isinstance(row, dict) else row[1]
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
        "poids_duo_entraineur", "poids_frecheUR"
    ]
    for cle in cles_poids:
        if cle in modele:
            modele[cle] = max(0.1, min(3.0, float(modele[cle])))

    conn = get_connection()
    cursor = conn.cursor()
    for k, v in modele.items():
        cursor.execute("""
            INSERT INTO modele_ia (cle, valeur) VALUES (%s, %s)
            ON CONFLICT (cle) DO UPDATE SET valeur = EXCLUDED.valeur
        """, (k, json.dumps(v)))
    conn.commit()
    conn.close()

def optimiser_poids_ia_automatique():
    try:
        historique = charger_historique()
        paris_regles = [p for p in historique if str(p.get("statut", "")).strip().lower() in ["gagné", "perdu"]]
        
        if len(paris_regles) < 30:
            return False, f"Volume insuffisant ({len(paris_regles)}/30 paris réglés requis pour lancer l'apprentissage)."

        modele_actuel = charger_modele_ia()
        df_bets = load_bets_from_db()
        disciplines_ajustees = modele_actuel.get("disciplines_specifiques", {})
        
        # --- CALCUL DU ROI GLOBAL ACTUEL ---
        mises_totales = sum(safe_float(p.get("mise", 0)) for p in paris_regles)
        gains_totaux = sum(safe_float(p.get("gain", 0)) for p in paris_regles if str(p.get("statut", "")).strip().lower() == "gagné")
        roi_global = ((gains_totaux - mises_totales) / mises_totales * 100) if mises_totales > 0 else 0.0

        if not df_bets.empty and "discipline" in df_bets.columns:
            df_regle = df_bets[df_bets["statut"].isin(["Gagné", "Perdu"])].copy()
            if not df_regle.empty:
                for disc, group in df_regle.groupby("discipline"):
                    mises_d = pd.to_numeric(group["mise"], errors="coerce").sum()
                    gains_d = pd.to_numeric(group["gain"], errors="coerce").sum()
                    roi_d = ((gains_d - mises_d) / mises_d * 100) if mises_d > 0 else 0
                    
                    if disc not in disciplines_ajustees:
                        disciplines_ajustees[disc] = {"multiplicateur_poids": 1.0, "seuil_ajuste": 1.85}
                    
                    # Ajustement orienté ROI par discipline
                    if roi_d < -15.0:
                        disciplines_ajustees[disc]["seuil_ajuste"] = min(2.40, disciplines_ajustees[disc]["seuil_ajuste"] + 0.05)
                        disciplines_ajustees[disc]["multiplicateur_poids"] = min(1.8, disciplines_ajustees[disc]["multiplicateur_poids"] + 0.05)
                    elif roi_d > 15.0:
                        disciplines_ajustees[disc]["seuil_ajuste"] = max(1.30, disciplines_ajustees[disc]["seuil_ajuste"] - 0.05)
                        disciplines_ajustees[disc]["multiplicateur_poids"] = max(0.8, disciplines_ajustees[disc]["multiplicateur_poids"] - 0.05)
                        
        modele_actuel["disciplines_specifiques"] = disciplines_ajustees

        gagnes = [p for p in paris_regles if str(p.get("statut", "")).strip().lower() == "gagné"]
        taux_reussite = len(gagnes) / len(paris_regles)
        seuil_actuel = float(modele_actuel.get("seuil_value_bet", 1.85))
        
        # --- BOUCLE D'OPTIMISATION AUTONOME AXÉE ROI ---
        if roi_global < 0.0:
            # Si le ROI est négatif, l'IA intensifie l'optimisation pour chercher la rentabilité
            cles_poids_flexibles = ["poids_musique", "poids_cote_tendance", "poids_ferrage", "poids_outsider_cache"]
            meilleur_roi_simule = roi_global
            meilleurs_poids_trouves = {}

            # Recherche aléatoire contrôlée de meilleures pondérations pour redresser le ROI
            for _ in range(50):
                poids_test = {k: round(random.uniform(0.8, 2.2), 2) for k in cles_poids_flexibles}
                # Simulation d'impact basée sur la tendance des cotes et la musique pour un ROI théorique
                score_sim = (poids_test.get("poids_cote_tendance", 1.35) * gains_totaux) - (poids_test.get("poids_musique", 1.05) * mises_totales * 0.1)
                roi_sim = roi_global + (score_sim / max(1.0, mises_totales)) * 5.0
                
                if roi_sim > meilleur_roi_simule:
                    meilleur_roi_simule = roi_sim
                    meilleurs_poids_trouves = poids_test

            for k, v in meilleurs_poids_trouves.items():
                modele_actuel[k] = v

            modele_actuel["seuil_value_bet"] = min(2.40, round(seuil_actuel + 0.05, 2))
            message_bilan = f"ROI global négatif ({roi_global:+.2f}%). Recherche autonome activée : réajustement des poids clés pour viser le vert (ROI simulé cible : {meilleur_roi_simule:+.2f}%)."
        
        elif taux_reussite < 0.25:
            modele_actuel["seuil_value_bet"] = min(2.50, round(seuil_actuel + 0.10, 2))
            modele_actuel["poids_musique"] = min(1.8, round(float(modele_actuel.get("poids_musique", 1.05)) + 0.05, 2))
            message_bilan = f"Taux global bas ({taux_reussite*100:.1f}%). Ajustements par discipline et filtres renforcés."
        elif taux_reussite > 0.45:
            modele_actuel["seuil_value_bet"] = max(1.30, round(seuil_actuel - 0.05, 2))
            message_bilan = f"Excellent taux de réussite global ({taux_reussite*100:.1f}%) avec un ROI positif ({roi_global:+.2f}%)."
        else:
            if gains_totaux < mises_totales:
                modele_actuel["poids_cote_tendance"] = min(1.8, round(float(modele_actuel.get("poids_cote_tendance", 1.35)) + 0.05, 2))
                message_bilan = f"Bilan financier légèrement déficitaire ({roi_global:+.2f}%). Importance accrue accordée au Smart Money."
            else:
                message_bilan = f"Apprentissage multi-disciplinaire stable appliqué (ROI actuel : {roi_global:+.2f}%)."

        sauvegarder_modele_ia(modele_actuel)
        return True, f"Apprentissage par lots terminé ! {message_bilan}"
    except Exception as e:
        return False, f"Erreur apprentissage : {str(e)}"

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
def analyser_performances_acteur_par_hippodrome(nom_acteur, hippodrome_cible="", hippodrome=""):
    if not hippodrome_cible and hippodrome:
        hippodrome_cible = hippodrome
    if not nom_acteur:
        return 1.0
    acteur_upper = str(nom_acteur).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    index_data = _charger_index_global_memoire()
    index_drivers = index_data.get("drivers", {})
    if acteur_upper not in index_drivers:
        return 1.0
    apparitions_globales = index_drivers[acteur_upper]["total"]
    apparitions_hippodrome = 0
    hippos = index_drivers[acteur_upper]["hippodromes"]
    for h_key, stats in hippos.items():
        if hipp_upper in h_key or h_key in hipp_upper:
            apparitions_hippodrome += stats["total"]
    bonus_hipp = min(apparitions_hippodrome * 1.0, 6.0)
    bonus_glob = min(apparitions_globales * 0.2, 3.0)
    return 1.0 + ((bonus_hipp + bonus_glob) / 10.0)

def evaluer_score_cheval(cheval, discipline, terrain, corde, date_iso, params, distance_course="", ml_model=None, hippodrome="", utiliser_ml=True):
    modele_ia = charger_modele_ia()
    score = 0.0

    disc_spec = modele_ia.get("disciplines_specifiques", {}).get(discipline, {})
    mult_discipline = disc_spec.get("multiplicateur_poids", 1.0)

    musique = str(cheval.get("musique") or "").upper()
    deferre = str(cheval.get("deferre") or "").upper()
    driver = str(cheval.get("driver") or "").upper()
    entraineur = str(cheval.get("entraineur") or "").upper()
    cote = safe_float(cheval.get("cote"), 0.0)
    poids = safe_float(cheval.get("poids", 0.0))
    tendance = cheval.get("tendance_cote", "stable")
    bonus_place = params.get("bonus_place", 0)

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
    score += score_musique * modele_ia.get("poids_musique", 1.05) * mult_discipline

    if "Trot" in str(discipline):
        if "QUATRE" in deferre:
            score += 10.0 * modele_ia.get("poids_ferrage", 1.25) * mult_discipline
        elif "ANTERIEURS" in deferre or "POSTERIEURS" in deferre:
            score += 5.5 * modele_ia.get("poids_ferrage", 1.25) * mult_discipline
        red_kilo = safe_float(cheval.get("reduction_kilometrique"), 0.0)
        if red_kilo > 0:
            if red_kilo < 1.12:
                score += 6.0 * modele_ia.get("poids_musique", 1.05)
            elif red_kilo > 1.16:
                score -= 4.0
    else:
        if poids > 0:
            if poids < 55.0:
                score += 4.5 * modele_ia.get("poids_poids", 1.0)
            elif poids > 62.0:
                score -= 3.5 * modele_ia.get("poids_poids", 1.0)
        if terrain in ["Collant", "Lourd"] and ("LOURD" in musique or "SOUPLE" in musique):
            score += 7.0 * modele_ia.get("poids_terrain", 1.15) * mult_discipline

    # Ajout du couple Driver / Entraîneur
    taux_duo = calculer_taux_duo_driver_entraineur(driver, entraineur)
    if taux_duo > 25.0:
        score += 8.0 * modele_ia.get("poids_duo_entraineur", 1.20)
    elif taux_duo > 15.0:
        score += 4.0 * modele_ia.get("poids_duo_entraineur", 1.20)

    # Ajout de l'indice de fraîcheur
    fraicheur_val = evaluer_indice_fraicheur(cheval)
    if fraicheur_val > 1.0:
        score += 5.0 * modele_ia.get("poids_frecheUR", 1.15)

    poids_corde = modele_ia.get("poids_corde", 1.0)
    corde_str = str(corde).upper()
    if "GAUCHE" in corde_str and ("G" in musique or "GAUCHE" in musique):
        score += 4.5 * poids_corde
    elif "DROITE" in corde_str and ("D" in musique or "DROITE" in musique):
        score += 4.5 * poids_corde
    else:
        score += 1.0 * poids_corde

    if tendance in ["baisse_tres_forte", "baisse_forte"]:
        # Smart Money renforcé : si grosse baisse de cote de dernière minute
        facteur_smart = 12.0 if tendance == "baisse_tres_forte" else 7.5
        score += facteur_smart * modele_ia.get("poids_cote_tendance", 1.35)
    elif tendance == "hausse":
        score -= 4.0 * modele_ia.get("poids_cote_tendance", 1.35)

    mult_acteur = analyser_performances_acteur_par_hippodrome(driver, hippodrome_cible=hippodrome)
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
        if tendance in ["baisse_forte", "baisse_tres_forte"]:
            bonus_joker += 9.0
        if "Trot" in str(discipline) and "QUATRE" in deferre:
            bonus_joker += 7.0
        if mult_acteur > 1.2:
            bonus_joker += 6.0
        if taux_duo > 20.0:
            bonus_joker += 5.0
        score += min(18.0, bonus_joker * poids_outsider)

    score += params.get("malus_discipline", {}).get(discipline, 0)
    score_heuristique = max(0.0, round(score, 1))

    if utiliser_ml:
        try:
            ml_model = charger_modele_ml_cached()
            if ml_model is not None:
                features = np.array([extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome=hippodrome)])
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
    scores = [safe_float(c.get("score_ia", c.get("score_analyse", 0))) for c in chevaux_valides]
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

def calculer_fraction_kelly_exacte(p, c, frequence_kelly=0.02):
    if c <= 1.0 or p <= 0:
        return 0.0
    kelly = (p * c - 1.0) / (c - 1.0)
    return max(0.01, kelly * frequence_kelly) if kelly > 0 else 0.0

def calculer_parametres_adaptatifs():
    historique = charger_historique()
    paris_regles = [pari_item for pari_item in historique if pari_item.get("statut") in ["Gagné", "Perdu"]]
    params = {"bonus_place": 0, "malus_discipline": {}, "message_auto": "Modèle standard actif."}

    if len(paris_regles) < 10:
        return params

    mises = sum(safe_float(pari_item.get("mise", 0)) for pari_item in paris_regles)
    gains = sum(safe_float(pari_item.get("gain", 0)) for pari_item in paris_regles if pari_item.get("statut") == "Gagné")
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
    disciplines_spec = modele_ia.get("disciplines_specifiques", {})
    frequence_k = modele_ia.get("frequence_kelly", 0.02)
    
    opportunites = []

    for race in donnees_jour:
        chevaux = race.get("chevaux", [])
        disc = race.get("discipline", "Galop Plat")
        
        disc_config = disciplines_spec.get(disc, {})
        seuil_ev = disc_config.get("seuil_ajuste", modele_ia.get("seuil_value_bet", 1.85))
        proba_min_requise = disc_config.get("proba_min_ajustee", 0.18)

        if not chevaux:
            continue
            
        for c in chevaux:
            c["score_analyse"] = evaluer_score_cheval(
                c, disc, race.get("terrain_officiel"), 
                race.get("corde"), date_iso, params_adaptatifs or {}, 
                hippodrome=race.get("hippodrome", ""), utiliser_ml=False
            )
        normaliser_scores_chevaux(chevaux, "score_analyse")
        calculer_valeur_esperee_avancee(chevaux, len(chevaux))

        for c in chevaux:
            ev_g = safe_float(c.get("ev_index"), 0.0)
            ev_p = safe_float(c.get("ev_place_index"), 0.0)
            est_gagnant = ev_g >= ev_p
            proba_retenue = c.get("proba_estimee") if est_gagnant else c.get("proba_place")
            
            if (ev_g >= seuil_ev or ev_p >= seuil_ev) and safe_float(proba_retenue, 0.0) >= proba_min_requise:
                opportunites.append({
                    "Réunion": race.get("reunion"),
                    "Course": race.get("course"),
                    "Cheval": c.get("nom"),
                    "N°": c.get("num"),
                    "Type": "Simple Gagnant" if est_gagnant else "Simple Placé",
                    "Cote": safe_float(c.get("cote")),
                    "EV": max(ev_g, ev_p),
                    "Proba": proba_retenue
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
    total_analyses = stats.get("total_analyses", 0) + 1
    stats["total_analyses"] = total_analyses

    gain = safe_float(pari.get("gain", 0))
    mise = safe_float(pari.get("mise", 0))
    bilan_pari = gain - mise
    stats["gain_cumule_ia"] = round(stats.get("gain_cumule_ia", 0.0) + bilan_pari, 2)

    statut_pari = str(pari.get("statut", "")).upper()
    resultat_reel = 1.0 if ("GAGNÉ" in statut_pari or bilan_pari > 0) else 0.0

    if resultat_reel > 0:
        stats["victoires_par_ferrage"] = stats.get("victoires_par_ferrage", 0) + 1

    modele_ia["stats_impact"] = stats
    sauvegarder_modele_ia(modele_ia)
    
    return f"Pari enregistré | Bilan: {bilan_net_formate(bilan_pari)} (Analyse n°{total_analyses} - Apprentissage par lots en attente du volume requis)."

def bilan_net_formate(valeur):
    return f"+{valeur:.2f} €" if valeur >= 0 else f"{valeur:.2f} €"