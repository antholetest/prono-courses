# ai_model.py
import json
import logging
import math
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

from utils.database import (
    DB_PATH,
    get_connection,
    charger_historique,
    charger_courses_jour_db,
    load_bets_from_db,
    save_bet_to_db,
)
from utils.helpers import safe_float

logger = logging.getLogger("PMU_Pro")


MODELE_IA_DEFAUT = {
    "poids_musique": 1.25,
    "poids_ferrage": 1.30,
    "poids_terrain": 1.20,
    "poids_poids": 1.0,
    "poids_cote_tendance": 1.35,
    "poids_driver": 1.20,
    "poids_corde": 1.0,
    "poids_hippodrome_acteur": 1.30,
    "poids_distance": 1.15,
    "poids_outsider_cache": 1.10,
    "poids_duo_entraineur": 1.25,
    "poids_frecheUR": 1.20,
    "seuil_value_bet": 1.30,
    "seuil_value_gagnant": 1.15,
    "seuil_value_place": 1.25,
    "frequence_kelly": 0.01,
    "version_modele": 3,
    "disciplines_specifiques": {
        "Trot Attelé": {
            "poids_ferrage": 1.50,
            "poids_musique": 1.50,
            "poids_driver": 1.50,
            "multiplicateur_poids": 0.90,
            "seuil_ev_gagnant": 1.15, 
            "seuil_ev_place": 1.25,
            "min_cote": 2.20,          
            "max_cote": 8.00,
            "bloque_paris": False,
        },
        "Trot Monté": {
            "poids_ferrage": 1.50,
            "poids_musique": 1.50,
            "poids_driver": 1.50,
            "multiplicateur_poids": 0.90,
            "seuil_ev_gagnant": 1.15,
            "seuil_ev_place": 1.25,
            "min_cote": 2.50,
            "max_cote": 8.00,
            "bloque_paris": False,
        },
        "Galop Plat": {
            "poids_terrain": 1.50,
            "poids_poids": 1.50,
            "poids_corde": 1.60,
            "multiplicateur_poids": 0.90,
            "seuil_ev_gagnant": 1.15,
            "seuil_ev_place": 1.25,
            "min_cote": 2.50,
            "max_cote": 8.00,
            "bloque_paris": False,
        },
        "Haies": {
            "poids_terrain": 1.70,
            "poids_distance": 1.60,
            "multiplicateur_poids": 0.85,
            "seuil_ev_gagnant": 1.20,  
            "seuil_ev_place": 1.30,
            "min_cote": 2.20,         
            "max_cote": 9.00,
            "bloque_paris": False,
        },
        "Steeple-chase": {
            "poids_terrain": 1.70,
            "poids_distance": 1.50,
            "multiplicateur_poids": 0.85,
            "seuil_ev_gagnant": 1.20,
            "seuil_ev_place": 1.30,
            "min_cote": 2.20,         
            "max_cote": 9.00,
            "bloque_paris": False,
        },
    },
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


# ============================================================
# INDEX GLOBAL / HISTORIQUE (OPTIMISÉ STREAMLIT CACHE)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _charger_index_global_memoire(date_max=None):
    global _CACHE_INDEX_GLOBAL
    if _CACHE_INDEX_GLOBAL is not None and date_max is None:
        return _CACHE_INDEX_GLOBAL

    index_global = {
        "drivers": {},
        "entraineurs": {},
        "duos": {},
        "hippodromes": {},
        "chevaux": {},
    }

    try:
        conn = get_connection()
        cursor = conn.cursor()

        if date_max:
            cursor.execute(
                """
                SELECT data_json
                FROM courses_cache
                WHERE date_iso < %s
                  AND date_iso >= TO_CHAR(%s::date - INTERVAL '30 days', 'YYYY-MM-DD')
                """,
                (date_max, date_max),
            )
        else:
            cursor.execute(
                """
                SELECT data_json
                FROM courses_cache
                WHERE date_iso >= TO_CHAR(CURRENT_DATE - INTERVAL '30 days', 'YYYY-MM-DD')
                """
            )

        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            val_json = row["data_json"] if isinstance(row, dict) else row[0]
            try:
                races = json.loads(val_json)
                if isinstance(races, dict):
                    races = [races]
                if not isinstance(races, list):
                    continue

                for race in races:
                    hipp_race = str(race.get("hippodrome", "")).upper().strip()
                    discipline = str(race.get("discipline", "")).upper().strip()
                    terrain = str(race.get("terrain_officiel", "")).upper().strip()

                    if hipp_race:
                        if hipp_race not in index_global["hippodromes"]:
                            index_global["hippodromes"][hipp_race] = {
                                "total_courses": 0,
                                "disciplines": set(),
                                "terrains": set(),
                            }
                        index_global["hippodromes"][hipp_race]["total_courses"] += 1
                        if discipline:
                            index_global["hippodromes"][hipp_race]["disciplines"].add(discipline)
                        if terrain:
                            index_global["hippodromes"][hipp_race]["terrains"].add(terrain)

                    for part in race.get("chevaux", []):
                        nom_cheval = str(part.get("nom", "")).upper().strip()
                        driver = str(part.get("driver", "")).upper().strip()
                        entraineur = str(part.get("entraineur", "")).upper().strip()
                        ordre = part.get("ordreArrivee", 0)

                        if nom_cheval:
                            if nom_cheval not in index_global["chevaux"]:
                                index_global["chevaux"][nom_cheval] = {
                                    "total": 0,
                                    "victoires": 0,
                                    "podiums": 0,
                                    "hippodromes": {},
                                    "terrains": {},
                                }
                            index_global["chevaux"][nom_cheval]["total"] += 1
                            if ordre == 1:
                                index_global["chevaux"][nom_cheval]["victoires"] += 1
                            if ordre in [1, 2, 3]:
                                index_global["chevaux"][nom_cheval]["podiums"] += 1

                            if hipp_race:
                                if hipp_race not in index_global["chevaux"][nom_cheval]["hippodromes"]:
                                    index_global["chevaux"][nom_cheval]["hippodromes"][hipp_race] = {
                                        "total": 0,
                                        "victoires": 0,
                                    }
                                index_global["chevaux"][nom_cheval]["hippodromes"][hipp_race]["total"] += 1
                                if ordre == 1:
                                    index_global["chevaux"][nom_cheval]["hippodromes"][hipp_race]["victoires"] += 1

                            if terrain:
                                if terrain not in index_global["chevaux"][nom_cheval]["terrains"]:
                                    index_global["chevaux"][nom_cheval]["terrains"][terrain] = {
                                        "total": 0,
                                        "victoires": 0,
                                    }
                                index_global["chevaux"][nom_cheval]["terrains"][terrain]["total"] += 1
                                if ordre == 1:
                                    index_global["chevaux"][nom_cheval]["terrains"][terrain]["victoires"] += 1

                        if driver and driver != "INCONNU":
                            if driver not in index_global["drivers"]:
                                index_global["drivers"][driver] = {
                                    "total": 0,
                                    "victoires": 0,
                                    "hippodromes": {},
                                }
                            index_global["drivers"][driver]["total"] += 1
                            if ordre == 1:
                                index_global["drivers"][driver]["victoires"] += 1

                            if hipp_race:
                                if hipp_race not in index_global["drivers"][driver]["hippodromes"]:
                                    index_global["drivers"][driver]["hippodromes"][hipp_race] = {
                                        "total": 0,
                                        "victoires": 0,
                                    }
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

    if date_max is None:
        _CACHE_INDEX_GLOBAL = index_global
    return index_global


def reinitialiser_index_memoire():
    global _CACHE_INDEX_GLOBAL
    _CACHE_INDEX_GLOBAL = None
    try:
        _charger_index_global_memoire.clear()
    except Exception:
        pass


# ============================================================
# FORMES / PERFORMANCE
# ============================================================
@st.cache_data(show_spinner=False)
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
            score_forme += 12.0 * poids
            victoires += 1
            podiums += 1
        elif token in ["2", "3"]:
            score_forme += 7.0 * poids
            podiums += 1
        elif token in ["4", "5"]:
            score_forme += 2.0 * poids
        elif token in ["D", "T", "A", "0"]:
            score_forme -= 6.0 * poids
            if token in ["D", "T", "A"]:
                fautes += 1

    return round(score_forme, 2), victoires, podiums, fautes


@st.cache_data(ttl=3600, show_spinner=False)
def calculer_taux_driver_hippodrome_db(nom_driver, hippodrome_cible, date_max=None):
    if not nom_driver or not hippodrome_cible:
        return 0.0

    driver_upper = str(nom_driver).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    index_data = _charger_index_global_memoire(date_max=date_max)
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


@st.cache_data(ttl=3600, show_spinner=False)
def calculer_taux_duo_driver_entraineur(nom_driver, nom_entraineur, date_max=None):
    if not nom_driver or not nom_entraineur:
        return 0.0

    duo_key = f"{str(nom_driver).upper().strip()}__{str(nom_entraineur).upper().strip()}"
    index_data = _charger_index_global_memoire(date_max=date_max)
    duos = index_data.get("duos", {})

    if duo_key not in duos:
        return 0.0

    stats = duos[duo_key]
    if stats["total"] < 3:
        return 0.0
    return round((stats["victoires"] / stats["total"]) * 100.0, 2)


@st.cache_data(ttl=3600, show_spinner=False)
def calculer_affinite_cheval_hippodrome(nom_cheval, hippodrome_cible, date_max=None):
    if not nom_cheval or not hippodrome_cible:
        return 1.0

    cheval_upper = str(nom_cheval).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    index_data = _charger_index_global_memoire(date_max=date_max)
    index_chevaux = index_data.get("chevaux", {})

    if cheval_upper not in index_chevaux:
        return 1.0

    hippos = index_chevaux[cheval_upper]["hippodromes"]
    total_c, total_v = 0, 0
    for h_key, stats in hippos.items():
        if hipp_upper in h_key or h_key in hipp_upper:
            total_c += stats["total"]
            total_v += stats["victoires"]

    if total_c > 0:
        return round(1.0 + (total_v / total_c) * 0.5, 2)
    return 1.0


@st.cache_data(ttl=3600, show_spinner=False)
def calculer_affinite_terrain(nom_cheval, terrain_cible, date_max=None):
    """Calcule l'affinité historique d'un cheval avec le terrain."""
    if not nom_cheval or not terrain_cible:
        return 1.0

    cheval_upper = str(nom_cheval).upper().strip()
    terrain_upper = str(terrain_cible).upper().strip()

    index_data = _charger_index_global_memoire(date_max=date_max)
    index_chevaux = index_data.get("chevaux", {})
    if cheval_upper not in index_chevaux:
        return 1.0

    terrains_dict = index_chevaux[cheval_upper].get("terrains", {})
    total_c, total_v = 0, 0

    for t_key, stats in terrains_dict.items():
        if terrain_upper in t_key or t_key in terrain_upper:
            total_c += stats["total"]
            total_v += stats["victoires"]

    if total_c >= 2:
        taux_reussite = total_v / total_c
        return round(1.0 + (taux_reussite * 0.8), 2)

    return 1.0


def evaluer_indice_fraicheur(cheval):
    musique = str(cheval.get("musique", "")).upper()
    if not musique:
        return 1.0

    tokens = re.findall(r"[1-90DATA]", musique)
    if len(tokens) >= 2:
        premier_token = tokens[0]
        if premier_token in ["D", "0"] and len(tokens) > 1 and tokens[1] in ["1", "2", "3"]:
            return 1.20
    return 1.0


# ============================================================
# EXTRACTION DE FEATURES
# ============================================================
def extraire_caracteristiques_cheval(cheval, discipline, terrain, hippodrome="", type_course="", date_max=None):
    score_forme, nb_victoires, nb_podiums, fautes = analyser_forme_musique_avancee(cheval.get("musique", ""))
    taux_driver_hipp = calculer_taux_driver_hippodrome_db(cheval.get("driver", ""), hippodrome, date_max=date_max)
    taux_duo = calculer_taux_duo_driver_entraineur(cheval.get("driver", ""), cheval.get("entraineur", ""), date_max=date_max)
    affinite_hipp_cheval = calculer_affinite_cheval_hippodrome(cheval.get("nom", ""), hippodrome, date_max=date_max)
    affinite_terrain = calculer_affinite_terrain(cheval.get("nom", ""), terrain, date_max=date_max)

    cote = safe_float(cheval.get("cote", cheval.get("cote_actuelle", 10.0)), 10.0)
    poids = safe_float(cheval.get("poids"), 0.0)
    poids_precedent = safe_float(cheval.get("poids_precedent", poids), poids)
    cart_poids = round(poids - poids_precedent, 2)

    red_kilo = safe_float(cheval.get("reduction_kilometrique"), 0.0)
    vitesse_estimee = round(100.0 / red_kilo, 2) if red_kilo > 0 else 0.0

    deferre_str = str(cheval.get("deferre", "")).upper()
    is_deferre_4 = 1 if "QUATRE" in deferre_str else 0

    tendance = str(cheval.get("tendance_cote", "stable"))
    tendance_val = 1.0 if "baisse" in tendance else (-1.0 if "hausse" in tendance else 0.0)

    is_trot = 1 if "Trot" in str(discipline) else 0
    is_lourd = 1 if terrain in ["Collant", "Lourd"] else 0
    affinite_terrain_lourd = round(affinite_terrain * is_lourd, 2)

    fraicheur = evaluer_indice_fraicheur(cheval)
    is_handicap = 1 if "HANDICAP" in str(type_course).upper() else 0

    return [
        score_forme,
        nb_victoires,
        nb_podiums,
        fautes,
        taux_driver_hipp,
        taux_duo,
        affinite_hipp_cheval,
        affinite_terrain,
        cote,
        poids,
        cart_poids,
        vitesse_estimee,
        is_deferre_4,
        tendance_val,
        is_trot,
        is_lourd,
        fraicheur,
        is_handicap,
        affinite_terrain_lourd,
    ]


# ============================================================
# ML / LIGHTGBM
# ============================================================
def _entrainer_modele_ml_depuis_db_impl(date_max=None):
    """Entraîne un classifieur LightGBM (LGBMClassifier) optimisé sans fuite de données."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if date_max:
            cursor.execute(
                """
                SELECT data_json
                FROM courses_cache
                WHERE date_iso < %s
                ORDER BY date_iso DESC
                LIMIT 500
                """,
                (date_max,),
            )
        else:
            cursor.execute(
                """
                SELECT data_json
                FROM courses_cache
                ORDER BY date_iso DESC
                LIMIT 500
                """
            )

        rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"Erreur lecture base pour ML : {e}")
        return None
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    X_all, y_all = [], []
    for row in rows:
        try:
            data_json = row["data_json"] if isinstance(row, dict) else row[0]
            races = json.loads(data_json)
            if isinstance(races, dict):
                races = [races]
            if not isinstance(races, list):
                continue

            for race in races:
                discipline = race.get("discipline", "Galop Plat")
                terrain = race.get("terrain_officiel", "Bon (Standard)")
                hippodrome = race.get("hippodrome", "")
                type_course = race.get("type_course", "")
                chevaux = race.get("chevaux", [])

                for part in chevaux:
                    cote = safe_float(part.get("cote", part.get("cote_actuelle", 0.0)), 0.0)
                    if cote <= 1.0:
                        continue

                    features = extraire_caracteristiques_cheval(
                        part,
                        discipline,
                        terrain,
                        hippodrome,
                        type_course,
                        date_max=date_max,
                    )
                    ordre = part.get("ordreArrivee", 0)
                    label = 1 if ordre == 1 else 0
                    X_all.append(features)
                    y_all.append(label)

        except Exception:
            continue

    if len(X_all) < 20:
        return None

    try:
        # Configuration optimisée pour les courses hippiques (gestion du déséquilibre + régularisation)
        classifier = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=5,            # Limite la profondeur pour éviter l'overfitting
            num_leaves=31,          # Contrôle la complexité des arbres
            is_unbalance=True,      # Gère le fait qu'il y ait beaucoup plus de perdants que de gagnants
            random_state=42,
            verbose=-1,
        )
        classifier.fit(np.array(X_all), np.array(y_all))

        model_bytes = pickle.dumps(classifier)
        
        # Sauvegarde sécurisée en base
        conn_save = get_connection()
        cur_save = conn_save.cursor()
        cur_save.execute(
            """
            INSERT INTO modele_ia (cle, valeur)
            VALUES ('lgb_model_serialized', %s)
            ON CONFLICT (cle) DO UPDATE SET valeur = EXCLUDED.valeur
            """,
            (model_bytes.hex(),),
        )
        conn_save.commit()
        cur_save.close()
        conn_save.close()

        return classifier

    except Exception as e:
        logger.error(f"Erreur entraînement LGBMClassifier : {e}")
        return None


@st.cache_resource
def charger_modele_ml_cached(date_max=None):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT valeur FROM modele_ia WHERE cle = 'lgb_model_serialized'")
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row and date_max is None:
            model_bytes = bytes.fromhex(row[0] if isinstance(row, str) else row.get("valeur"))
            return pickle.loads(model_bytes)
    except Exception:
        pass

    return _entrainer_modele_ml_depuis_db_impl(date_max=date_max)


def rafraichir_modele_ml():
    try:
        charger_modele_ml_cached.clear()
    except Exception:
        pass
    return _entrainer_modele_ml_depuis_db_impl()


def predire_proba_victoire_ml(cheval, discipline, terrain, hippodrome="", type_course="", date_max=None):
    """Prédit la probabilité réelle de victoire via le classifieur ML."""
    model = charger_modele_ml_cached(date_max=date_max)
    if model is None:
        return None

    try:
        features = extraire_caracteristiques_cheval(
            cheval,
            discipline,
            terrain,
            hippodrome,
            type_course,
            date_max=date_max,
        )
        proba = model.predict_proba([features])[0][1]
        return float(proba)
    except Exception as e:
        logger.error(f"Erreur prédiction proba ML : {e}")
        return None


# ============================================================
# MODELE IA / CONFIGURATION
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
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

    if int(safe_float(modele.get("version_modele"), 0)) < 3:
        modele["disciplines_specifiques"] = {
            d: dict(cfg) for d, cfg in MODELE_IA_DEFAUT["disciplines_specifiques"].items()
        }
        modele["version_modele"] = 3
        sauvegarder_modele_ia(modele)

    for k, v in MODELE_IA_DEFAUT.items():
        if k not in modele:
            modele[k] = v

    if "disciplines_specifiques" in MODELE_IA_DEFAUT:
        if "disciplines_specifiques" not in modele:
            modele["disciplines_specifiques"] = {}
        for disc, conf_defaut in MODELE_IA_DEFAUT["disciplines_specifiques"].items():
            if disc not in modele["disciplines_specifiques"]:
                modele["disciplines_specifiques"][disc] = conf_defaut.copy()
            else:
                for sub_k, sub_v in conf_defaut.items():
                    if sub_k not in modele["disciplines_specifiques"][disc]:
                        modele["disciplines_specifiques"][disc][sub_k] = sub_v

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
        "poids_duo_entraineur",
        "poids_frecheUR",
    ]

    for cle in cles_poids:
        if cle in modele:
            try:
                modele[cle] = max(0.1, min(3.0, float(modele[cle])))
            except (ValueError, TypeError):
                pass

    conn = get_connection()
    cursor = conn.cursor()
    try:
        for k, v in modele.items():
            cursor.execute(
                """
                INSERT INTO modele_ia (cle, valeur)
                VALUES (%s, %s)
                ON CONFLICT (cle) DO UPDATE SET valeur = EXCLUDED.valeur
                """,
                (str(k), json.dumps(v, ensure_ascii=False)),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la sauvegarde du modèle IA : {e}")
    finally:
        cursor.close()
        conn.close()
    
    try:
        charger_modele_ia.clear()
    except Exception:
        pass


# ============================================================
# OPTIMISATION AI (IDEMPOTENTE, SANS CUMUL)
# ============================================================
def obtenir_pas_et_reglage_selon_roi(roi_val):
    """
    Ajuste la granulosité de l'IA selon le ROI sur [ -100 ; +100 ].
    """
    roi = max(-100.0, min(100.0, float(roi_val)))

    if roi <= -50.0:
        return {"pas_ajustement": 0.08, "delta_seuil_ev": 0.04, "mode": "URGENCE_CRITIQUE"}
    elif -50.0 < roi <= -20.0:
        return {"pas_ajustement": 0.04, "delta_seuil_ev": 0.02, "mode": "REDRESSEMENT"}
    elif -20.0 < roi <= 0.0:
        return {"pas_ajustement": 0.015, "delta_seuil_ev": 0.005, "mode": "AFFINEMENT_EQUILIBRE"}
    elif 0.0 < roi <= 30.0:
        return {"pas_ajustement": 0.005, "delta_seuil_ev": -0.005, "mode": "RENTABILITE_ACTIVE"}
    else:
        return {"pas_ajustement": 0.001, "delta_seuil_ev": -0.01, "mode": "MAITRISE_STABLE"}


def optimiser_poids_ia_automatique():
    try:
        df_bets = load_bets_from_db()
        if df_bets.empty:
            return False, "Aucun pari enregistré pour lancer l'apprentissage."

        df_regle = df_bets[df_bets["statut"].isin(["Gagné", "Perdu"])].copy()
        df_regle = df_regle[pd.to_numeric(df_regle["mise"], errors="coerce").fillna(0) > 0]

        if len(df_regle) < 5:
            return False, f"Volume insuffisant ({len(df_regle)}/5 paris réglés requis)."

        modele_actuel = charger_modele_ia()
        if "disciplines_specifiques" not in modele_actuel:
            modele_actuel["disciplines_specifiques"] = {}

        disciplines_ajustees = modele_actuel["disciplines_specifiques"]
        disciplines_connues = set(disciplines_ajustees.keys())

        if not df_regle.empty:
            disciplines_connues.update(df_regle["discipline"].dropna().unique())

        for disc in disciplines_connues:
            if not disc or str(disc).strip() == "":
                continue

            if disc not in disciplines_ajustees:
                disciplines_ajustees[disc] = {
                    "poids_musique": float(modele_actuel.get("poids_musique", 1.25)),
                    "poids_ferrage": float(modele_actuel.get("poids_ferrage", 1.30)),
                    "poids_terrain": float(modele_actuel.get("poids_terrain", 1.20)),
                    "poids_driver": float(modele_actuel.get("poids_driver", 1.20)),
                    "multiplicateur_poids": 1.0,
                    "seuil_ev_gagnant": 1.10,
                    "seuil_ev_place": 1.20,
                    "bloque_paris": False,
                }

            disc_cfg = disciplines_ajustees[disc]
            roi_d = 0.0
            nb_reels_d = 0

            if not df_regle.empty:
                group = df_regle[df_regle["discipline"] == disc]
                if not group.empty:
                    mises_d = pd.to_numeric(group["mise"], errors="coerce").sum()
                    gains_d = pd.to_numeric(group["gain"], errors="coerce").sum()
                    roi_d = ((gains_d - mises_d) / mises_d * 100) if mises_d > 0 else 0.0
                    nb_reels_d = len(group)

            try:
                base_cfg = MODELE_IA_DEFAUT["disciplines_specifiques"].get(disc, {})
                base_g = float(base_cfg.get("seuil_ev_gagnant", 1.10))
                base_p = float(base_cfg.get("seuil_ev_place", 1.20))

                if nb_reels_d >= 30:
                    decalage = max(-0.03, min(0.10, -roi_d / 100.0 * 0.15))
                    disc_cfg["seuil_ev_gagnant"] = round(base_g + decalage, 3)
                    disc_cfg["seuil_ev_place"] = round(base_p + decalage * 0.8, 3)
                    disc_cfg["multiplicateur_poids"] = round(
                        max(0.9, min(1.1, 1.0 + roi_d / 100.0 * 0.2)),
                        4,
                    )
                    disc_cfg["mode_palier_actif"] = obtenir_pas_et_reglage_selon_roi(roi_d)["mode"]
                else:
                    disc_cfg["seuil_ev_gagnant"] = base_g
                    disc_cfg["seuil_ev_place"] = base_p
                    disc_cfg["multiplicateur_poids"] = 1.0
                    disc_cfg["mode_palier_actif"] = f"COLLECTE ({nb_reels_d}/30 paris réels)"

            except Exception as e_palier:
                logger.error(f"Erreur calcul réglage pour {disc} : {e_palier}")

        modele_actuel["disciplines_specifiques"] = disciplines_ajustees
        sauvegarder_modele_ia(modele_actuel)

        try:
            rafraichir_modele_ml()
        except Exception:
            pass

        return True, "Apprentissage automatique réussi : les poids et seuils par discipline ont été ajustés."

    except Exception as e:
        return False, f"Erreur lors de l'apprentissage automatique : {str(e)}"


# ============================================================
# NORMALISATION / VALEUR ATTENDUE
# ============================================================
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

    match = re.search(r"(\d+)", str(distance_course))
    dist_val = float(match.group(1)) if match else 0.0
    musique = str(cheval.get("musique") or "").upper()

    if dist_val > 0:
        if "1" in musique[:4]:
            return 1.25
        elif "2" in musique[:4] or "3" in musique[:4]:
            return 1.12
    return 1.05


@st.cache_data(ttl=3600, show_spinner=False)
def analyser_performances_acteur_par_hippodrome(nom_acteur, hippodrome_cible="", hippodrome="", date_max=None):
    if not hippodrome_cible and hippodrome:
        hippodrome_cible = hippodrome
    if not nom_acteur:
        return 1.0, 0

    acteur_upper = str(nom_acteur).upper().strip()
    hipp_upper = str(hippodrome_cible).upper().strip()
    index_data = _charger_index_global_memoire(date_max=date_max)
    index_drivers = index_data.get("drivers", {})

    if acteur_upper not in index_drivers:
        return 1.0, 0

    apparitions_globales = index_drivers[acteur_upper]["total"]
    apparitions_hippodrome = 0
    hippos = index_drivers[acteur_upper]["hippodromes"]

    for h_key, stats in hippos.items():
        if hipp_upper in h_key or h_key in hipp_upper:
            apparitions_hippodrome += stats["total"]

    bonus_hipp = min(apparitions_hippodrome * 1.0, 6.0)
    bonus_glob = min(apparitions_globales * 0.2, 3.0)
    mult = 1.0 + ((bonus_hipp + bonus_glob) / 10.0)
    return mult, apparitions_hippodrome


# ============================================================
# SCORE IA CHEVAL
# ============================================================
def evaluer_score_cheval(cheval, discipline, terrain, corde, date_iso, params, distance_course="", ml_model=None, hippodrome="", utiliser_ml=True, modele_ia=None):
    if modele_ia is None:
        modele_ia = charger_modele_ia()

    score = 0.0
    disc_configs = modele_ia.get("disciplines_specifiques", {})
    disc_spec = disc_configs.get(discipline, {})

    mult_discipline = disc_spec.get("multiplicateur_poids", 1.0)

    p_musique = disc_spec.get("poids_musique", modele_ia.get("poids_musique", 1.15))
    p_ferrage = disc_spec.get("poids_ferrage", modele_ia.get("poids_ferrage", 1.30))
    p_terrain = disc_spec.get("poids_terrain", modele_ia.get("poids_terrain", 1.20))
    p_driver = disc_spec.get("poids_driver", modele_ia.get("poids_driver", 1.20))
    p_tendance = disc_spec.get("poids_cote_tendance", modele_ia.get("poids_cote_tendance", 1.40))
    p_poids = disc_spec.get("poids_poids", modele_ia.get("poids_poids", 1.0))
    p_corde = disc_spec.get("poids_corde", modele_ia.get("poids_corde", 1.0))

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
            score_musique += 14 if idx >= 3 else 16
        elif char == "2":
            score_musique += 8 + bonus_place
        elif char == "3":
            score_musique += 6 + bonus_place
        elif char in ["0", "D", "T", "A"]:
            score_musique -= 8
    score += score_musique * p_musique * mult_discipline

    if "Trot" in str(discipline):
        if "QUATRE" in deferre:
            score += 12.0 * p_ferrage * mult_discipline
        red_kilo = safe_float(cheval.get("reduction_kilometrique"), 0.0)
        if red_kilo > 0 and red_kilo < 1.12:
            score += 7.0
    else:
        if terrain in ["Collant", "Lourd"] and ("LOURD" in musique or "SOUPLE" in musique):
            score += 8.0 * p_terrain * mult_discipline
        if poids > 0:
            score += p_poids * 2.0

    taux_duo = calculer_taux_duo_driver_entraineur(driver, entraineur, date_max=date_iso)
    if taux_duo > 20.0:
        score += 9.0 * modele_ia.get("poids_duo_entraineur", 1.25)

    affinite_t = calculer_affinite_terrain(cheval.get("nom", ""), terrain, date_max=date_iso)

    if "Trot" not in str(discipline):
        if terrain in ["Collant", "Lourd"]:
            if affinite_t > 1.1:
                score += (affinite_t - 1.0) * 25.0 * p_terrain * mult_discipline
            elif affinite_t < 0.9:
                score -= 15.0 * p_terrain * mult_discipline

            if "LOURD" in musique or "SOUPLE" in musique:
                score += 12.0 * p_terrain * mult_discipline
        else:
            score += (affinite_t - 1.0) * 8.0 * p_terrain * mult_discipline

        if poids > 0:
            score += p_poids * 2.0

    res_acteur = analyser_performances_acteur_par_hippodrome(driver, hippodrome_cible=hippodrome, date_max=date_iso)
    if isinstance(res_acteur, tuple):
        mult_acteur, nb_courses_hipp = res_acteur
    else:
        mult_acteur, nb_courses_hipp = res_acteur, 0

    if nb_courses_hipp > 10:
        mult_acteur = 1.0 + (mult_acteur - 1.0) * 1.5

    score += (mult_acteur - 1.0) * 12.0 * p_driver

    if utiliser_ml:
        proba_ml = predire_proba_victoire_ml(cheval, discipline, terrain, hippodrome=hippodrome, date_max=date_iso)
        if proba_ml is not None and proba_ml > 0:
            score += proba_ml * 50.0 * p_tendance

    return max(0.0, round(score, 1))


def calculer_valeur_esperee_avancee(chevaux_valides, nb_partants=12):
    if not chevaux_valides:
        return chevaux_valides

    temperature = 30.0
    scores = [safe_float(c.get("score_ia", c.get("score_analyse", 0))) for c in chevaux_valides]
    max_score = max(scores, default=0.0)
    exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
    somme_exp = sum(exp_scores)
    nb_places = 3 if nb_partants >= 8 else 2

    for i, c in enumerate(chevaux_valides):
        proba_estimee = exp_scores[i] / somme_exp if somme_exp > 0 else (1.0 / nb_partants)
        c["proba_estimee"] = round(proba_estimee, 4)

        cote = safe_float(c.get("cote"), 0.0)
        c["ev_index"] = round(proba_estimee * cote, 2) if cote > 1.0 else 0.0

        proba_place = min(0.70, proba_estimee * 2.2 + (1.0 / nb_partants))
        c["proba_place"] = round(proba_place, 4)

        cote_place_reelle = c.get("cote_place_reelle")
        if cote_place_reelle and cote_place_reelle > 1.0:
            cote_place = cote_place_reelle
        else:
            cote_place = max(1.05, 1.0 + (cote - 1.0) / 5.0)

        c["cote_place_estimee"] = round(cote_place, 2)
        c["ev_place_index"] = round(proba_place * cote_place, 2) if cote > 1.0 else 0.0

    return chevaux_valides


# ============================================================
# KELLY / PLAN BUDGET
# ============================================================
def calculer_fraction_kelly_exacte(p, c, frequence_kelly=0.01, proba_estimee=0.0):
    if c <= 1.0 or p <= 0:
        return 0.0

    kelly = (p * c - 1.0) / (c - 1.0)
    multiplicateur_confiance = 1.2 if proba_estimee > 0.30 else (0.7 if proba_estimee < 0.12 else 1.0)
    return max(0.01, kelly * frequence_kelly * multiplicateur_confiance) if kelly > 0 else 0.0


def calculer_parametres_adaptatifs():
    historique = charger_historique()
    paris_regles = [p for p in historique if p.get("statut") in ["Gagné", "Perdu"] and safe_float(p.get("mise")) > 0]
    params = {"bonus_place": 0, "malus_discipline": {}, "message_auto": "Modèle adaptatif actif."}

    if len(paris_regles) < 5:
        return params

    mises = sum(safe_float(p.get("mise", 0)) for p in paris_regles)
    gains = sum(safe_float(p.get("gain", 0)) for p in paris_regles if p.get("statut") == "Gagné")
    roi_global = ((gains - mises) / mises * 100) if mises > 0 else 0.0

    if roi_global < -10.0:
        params["bonus_place"] = 3
        params["message_auto"] = f"ROI global faible ({roi_global:.1f}%) : bascule stricte vers la sécurité et les places."

    return params


def generer_plan_budget_journalier(date_iso, budget_total=50.0, params_adaptatifs=None):
    donnees_jour, _ = charger_courses_jour_db(date_iso)
    if not donnees_jour:
        return []

    modele_ia = charger_modele_ia()
    disciplines_spec = modele_ia.get("disciplines_specifiques", {})
    frequence_k = modele_ia.get("frequence_kelly", 0.01)

    seuil_gagnant_global = safe_float(modele_ia.get("seuil_value_gagnant"), 1.15)
    seuil_place_global = safe_float(modele_ia.get("seuil_value_place"), 1.10)

    opportunites = []

    for race in donnees_jour:
        chevaux = race.get("chevaux", [])
        if len(chevaux) < 10:
            continue

        nom_course_upper = str(race.get("nom_course", "")).upper()
        if "RECLAMER" in nom_course_upper or "RÉCLAMER" in nom_course_upper:
            continue

        disc = race.get("discipline", "Galop Plat")
        disc_config = disciplines_spec.get(disc, {})
        if disc_config.get("bloque_paris", False):
            continue

        if not chevaux:
            continue

        seuil_gagnant = safe_float(disc_config.get("seuil_ev_gagnant"), seuil_gagnant_global)
        seuil_place = safe_float(disc_config.get("seuil_ev_place"), seuil_place_global)

        for c in chevaux:
            c["score_analyse"] = evaluer_score_cheval(
                c,
                disc,
                race.get("terrain_officiel"),
                race.get("corde", "Corde standard"),
                date_iso,
                params_adaptatifs or {},
                hippodrome=race.get("hippodrome", ""),
                utiliser_ml=True,
                modele_ia=modele_ia,
            )

        normaliser_scores_chevaux(chevaux, "score_analyse")
        calculer_valeur_esperee_avancee(chevaux, len(chevaux))

        for c in chevaux:
            ev_g = safe_float(c.get("ev_index"), 0.0)
            ev_p = safe_float(c.get("ev_place_index"), 0.0)

            est_gagnant = ev_g >= ev_p
            proba_retenue = c.get("proba_estimee") if est_gagnant else c.get("proba_place")

            cote_actuelle = safe_float(c.get("cote"))
            cote_ouverture = safe_float(c.get("cote_ouverture", cote_actuelle))
            if cote_ouverture > 0 and (cote_actuelle / cote_ouverture) < 0.65:
                continue

            if cote_actuelle > 12.0:
                continue

            if est_gagnant and proba_retenue < 0.12:
                continue
            if not est_gagnant and proba_retenue < 0.22:
                continue

            seuil_requis = seuil_gagnant if est_gagnant else seuil_place

            if (est_gagnant and ev_g >= seuil_requis) or (not est_gagnant and ev_p >= seuil_requis):
                opportunites.append({
                    "Réunion": race.get("reunion"),
                    "Course": race.get("course"),
                    "Cheval": c.get("nom"),
                    "N°": c.get("num"),
                    "Type": "Simple Gagnant" if est_gagnant else "Simple Placé",
                    "Cote": cote_actuelle,
                    "EV": round(ev_g if est_gagnant else ev_p, 2),
                    "Proba": proba_retenue,
                })

    if not opportunites:
        return []

    df_opp = pd.DataFrame(opportunites)
    df_opp["kelly_f"] = df_opp.apply(
        lambda r: calculer_fraction_kelly_exacte(r["Proba"], r["Cote"], frequence_k, r["Proba"]),
        axis=1,
    )
    tot_f = df_opp["kelly_f"].sum()

    facteur_engagement = min(1.0, max(0.3, tot_f * 5.0)) if tot_f > 0 else 0.0

    if tot_f > 0 and facteur_engagement > 0.1:
        df_opp["Mise (€)"] = ((df_opp["kelly_f"] / tot_f) * (budget_total * facteur_engagement)).round(1)
    else:
        return []

    return df_opp[["Réunion", "Course", "N°", "Cheval", "Type", "Cote", "EV", "Mise (€)"]]


# ============================================================
# FEEDBACK / APPRENTISSAGE
# ============================================================
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
    est_gagnant = ("GAGNÉ" in statut_pari or bilan_pari > 0)

    if est_gagnant and partants_bruts:
        details_pari = str(pari.get("cheval", "")).upper()
        num_match = re.search(r"N°\s*(\d+)", details_pari)
        if num_match:
            num_gagnant = num_match.group(1)
            for part in partants_bruts:
                if str(part.get("numPmu")) == num_gagnant:
                    deferre_info = str(part.get("deferre", "")).upper()
                    if "QUATRE" in deferre_info or "POSTERIEURS" in deferre_info or "F" in deferre_info:
                        stats["victoires_par_ferrage"] = stats.get("victoires_par_ferrage", 0) + 1
                    break

    discipline_pari = pari.get("discipline", "Galop Plat")
    disciplines_spec = modele_ia.get("disciplines_specifiques", {})

    if discipline_pari not in disciplines_spec:
        disciplines_spec[discipline_pari] = {
            "multiplicateur_poids": 1.0,
            "seuil_ajuste": float(modele_ia.get("seuil_value_bet", 1.20)),
            "bloque_paris": False,
        }

    disc_cfg = disciplines_spec[discipline_pari]
    seuil_actuel_disc = float(disc_cfg.get("seuil_ajuste", 1.20))

    if not est_gagnant:
        disc_cfg["seuil_ajuste"] = min(2.00, round(seuil_actuel_disc + 0.05, 2))
        disc_cfg["multiplicateur_poids"] = max(0.8, round(disc_cfg.get("multiplicateur_poids", 1.0) - 0.05, 2))
    else:
        if seuil_actuel_disc > 1.10:
            disc_cfg["seuil_ajuste"] = max(1.10, round(seuil_actuel_disc - 0.02, 2))
            disc_cfg["multiplicateur_poids"] = min(1.5, round(disc_cfg.get("multiplicateur_poids", 1.0) + 0.02, 2))

    modele_ia["disciplines_specifiques"] = disciplines_spec
    modele_ia["stats_impact"] = stats
    sauvegarder_modele_ia(modele_ia)

    return f"Pari enregistré | Bilan: {bilan_net_formate(bilan_pari)} ({discipline_pari} - Seuil ajusté à : {disc_cfg['seuil_ajuste']})"


def bilan_net_formate(valeur):
    return f"+{valeur:.2f} €" if valeur >= 0 else f"{valeur:.2f} €"


# ============================================================
# SIMULATION JOURNÉE / APPRENTISSAGE
# ============================================================
def simuler_et_entrainer_journee_intelligente(date_iso, mise_defaut=10.0):
    from utils.api_pmu import telecharger_pmu_date

    donnees_jour, _ = charger_courses_jour_db(date_iso)
    if not donnees_jour:
        succes_dl = telecharger_pmu_date(date_iso, None, afficher_progres=False)
        if not succes_dl:
            return 0, 0.0
        donnees_jour, _ = charger_courses_jour_db(date_iso)

    if not donnees_jour:
        return 0, 0.0

    modele_ia = charger_modele_ia()
    disc_spec_map = modele_ia.get("disciplines_specifiques", {})

    params_ad = calculer_parametres_adaptatifs()
    nb_paris_ajoutes = 0
    mises_jour = 0.0

    for c_elem in donnees_jour:
        nom_c = str(c_elem.get("nom_course", "")).strip()
        if not nom_c or nom_c.isdigit() or len(nom_c) <= 2:
            continue

        r_nom = f"{c_elem.get('reunion', 'R1')} - {c_elem.get('hippodrome', 'HIPPODROME')}"
        discipline_course = c_elem.get("discipline", "Galop Plat")

        config_discipline = disc_spec_map.get(discipline_course, {})
        est_bloque = config_discipline.get("bloque_paris", False)
        seuil_ev_g = float(config_discipline.get("seuil_ev_gagnant", modele_ia.get("seuil_value_gagnant", 1.10)))
        seuil_ev_p = float(config_discipline.get("seuil_ev_place", modele_ia.get("seuil_value_place", 1.20)))

        chevaux_c = c_elem.get("chevaux", [])
        if not chevaux_c:
            continue

        for c in chevaux_c:
            c["score_analyse"] = evaluer_score_cheval(
                c,
                discipline_course,
                c_elem.get("terrain_officiel"),
                c_elem.get("corde", "Corde standard"),
                date_iso,
                params_ad,
                distance_course=c_elem.get("distance", ""),
                utiliser_ml=True,
                modele_ia=modele_ia,
            )

        normaliser_scores_chevaux(chevaux_c, "score_analyse")
        calculer_valeur_esperee_avancee(chevaux_c, len(chevaux_c))

        meilleur_sim = max(chevaux_c, key=lambda x: safe_float(x.get("ev_index", 0)), default=None)
        if meilleur_sim:
            num_cheval = meilleur_sim.get("num", "?")
            nom_cheval = meilleur_sim.get("nom", "Inconnu")
            cote_sim = safe_float(meilleur_sim.get("cote"), 0.0)
            ev_sim = max(safe_float(meilleur_sim.get("ev_index", 0)), safe_float(meilleur_sim.get("ev_place_index", 0)))
            details_sim = f"Simulation IA (Entraînement Cache) ➔ N°{num_cheval} {nom_cheval} (Cote: {cote_sim} | EV: {ev_sim:.2f})"

            save_bet_to_db(
                date=date_iso,
                reunion=r_nom,
                course=c_elem.get("course", "C1"),
                cheval=details_sim,
                cote=cote_sim,
                mise=0.0,
                statut="En attente",
                gain=0.0,
                discipline=discipline_course,
            )

        if est_bloque:
            continue

        min_c = float(config_discipline.get("min_cote", 2.20))
        max_c = float(config_discipline.get("max_cote", 8.50))
        chevaux_val_c = [c for c in chevaux_c if min_c <= safe_float(c.get("cote")) <= max_c]
        if not chevaux_val_c:
            continue

        candidats_valides = [
            c for c in chevaux_val_c
            if safe_float(c.get("ev_place_index", 0)) >= seuil_ev_p
            and c.get("tendance_cote") != "hausse"
        ]

        candidats_valides.sort(key=lambda x: safe_float(x.get("ev_place_index", 0)), reverse=True)

        if candidats_valides:
            b_chev = candidats_valides[0]
            num_b = b_chev.get("num", "?")
            nom_b = b_chev.get("nom", "Inconnu")
            cote_b = safe_float(b_chev.get("cote"), 0.0)
            ev_b = safe_float(b_chev.get("ev_place_index", 0))

            details_reel = f"Simple Placé ({discipline_course} Value Auto) ➔ N°{num_b} {nom_b} (Cote: {cote_b} | EV Placé: {ev_b:.2f})"
            mise_effective = float(mise_defaut)
            nb_paris_ajoutes += 1
            mises_jour += mise_effective

            save_bet_to_db(
                date=date_iso,
                reunion=r_nom,
                course=c_elem.get("course", "C1"),
                cheval=details_reel,
                cote=cote_b,
                mise=mise_effective,
                statut="En attente",
                gain=0.0,
                discipline=discipline_course,
            )

    return nb_paris_ajoutes, mises_jour