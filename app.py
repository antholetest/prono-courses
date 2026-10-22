import datetime
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --- IMPORTS INTERNES ---
from utils.database import (
    init_db,
    charger_historique,
    sauvegarder_historique,
    charger_courses_jour_db,
    sauvegarder_courses_jour_db,
    load_bets_from_db,
    update_bet_status,
    get_conn,
    save_bet_to_db,
    save_multiple_bets_to_db,
    get_bankroll,
    update_bankroll,
    analyser_roi_par_discipline_sql,
    analyser_bilan_journalier_sql,
    migrer_anciens_json_vers_sqlite,
)
from utils.helpers import safe_float
from utils.api_pmu import telecharger_pmu_date, HEADERS
from utils.ai_model import (
    charger_modele_ia,
    sauvegarder_modele_ia,
    calculer_parametres_adaptatifs,
    evaluer_score_cheval,
    normaliser_scores_chevaux,
    calculer_valeur_esperee_avancee,
    generer_plan_budget_journalier,
    retroaction_apprentissage_ia,
    rafraichir_modele_ml,
    optimiser_poids_ia_automatique,
    simuler_et_entrainer_journee_intelligente,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PMU_Pro")

st.set_page_config(
    page_title="Analyse & Stratégie PMU Pro + IA",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DOSSIER = Path(".")

# --- INITIALISATION DB ---
init_db()


# ============================================================
# CACHES POUR ÉVITER LES RELANCES INUTILES
# ============================================================
@st.cache_data(ttl=30)
def cached_charger_historique():
    return charger_historique()


@st.cache_data(ttl=30)
def cached_load_bets_from_db():
    return load_bets_from_db()


@st.cache_data(ttl=30)
def cached_charger_courses_jour_db(date_iso):
    return charger_courses_jour_db(date_iso)


@st.cache_data(ttl=60)
def cached_charger_modele_ia():
    return charger_modele_ia()


@st.cache_data(ttl=30)
def cached_calculer_parametres_adaptatifs():
    return calculer_parametres_adaptatifs()


# --- AUTHENTIFICATION ---
if "authentifie" not in st.session_state:
    st.session_state["authentifie"] = False

if not st.session_state["authentifie"]:
    st.title("🔒 Espace Restreint - Connexion Sécurisée")
    mot_de_passe_saisi = st.text_input("Entrez votre mot de passe", type="password")
    if st.button("Se connecter"):
        mdp_attendu = st.secrets.get("PASSWORD", "301180")
        if mot_de_passe_saisi.strip() == mdp_attendu:
            st.session_state["authentifie"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()


# ============================================================
# SAS UNIQUE : ENREGISTREMENT PARI VIA L'IA
# ============================================================
def enregistrer_pari_via_ia(date_iso, reunion, course_num, discipline, cheval_data, type_pari_souhaite, mise_totale):
    """
    Enregistre directement le pari en base sans recharger ni relire l'historique.
    """
    if not isinstance(cheval_data, dict):
        return False

    score_ia = safe_float(cheval_data.get("score_ia", cheval_data.get("score_analyse", 0.0)))
    ev_gagnant = safe_float(cheval_data.get("ev_index", 0.0))
    ev_place = safe_float(cheval_data.get("ev_place_index", 0.0))

    num_cheval = cheval_data.get("num", "?")
    nom_cheval = cheval_data.get("nom", "Inconnu")
    cote = safe_float(cheval_data.get("cote"), 0.0)

    max_ev = max(ev_gagnant, ev_place)
    details_str = f"{type_pari_souhaite} ➔ N°{num_cheval} {nom_cheval} (Cote: {cote} | EV: {max_ev:.2f})"

    save_bet_to_db(
        date=date_iso,
        reunion=reunion,
        course=course_num,
        cheval=details_str,
        cote=cote,
        mise=float(mise_totale),
        statut="En attente",
        gain=0.0,
        discipline=discipline,
    )
    return True


# ============================================================
# FONCTIONS STATISTIQUES / ROI
# ============================================================
def test_student_yield(paris_regles, mu_0=-0.15):
    rendements = []
    for pari_item in paris_regles:
        m = safe_float(pari_item.get("mise", 0))
        g = safe_float(pari_item.get("gain", 0))
        if m > 0:
            rendements.append((g - m) / m)

    if len(rendements) < 2:
        return None, None

    mean_r = sum(rendements) / len(rendements)
    variance = sum((r - mean_r) ** 2 for r in rendements) / (len(rendements) - 1)
    s = math.sqrt(variance)
    if s == 0:
        return None, None

    t_score = (mean_r - mu_0) / (s / math.sqrt(len(rendements)))
    return t_score, mean_r


def test_significativite_monte_carlo(paris_regles, iterations=10000):
    if len(paris_regles) < 10:
        return None, None

    profit_reel = sum(
        safe_float(pari_item.get("gain", 0)) - safe_float(pari_item.get("mise", 0))
        for pari_item in paris_regles
    )
    profits_nets = [
        safe_float(pari_item.get("gain", 0)) - safe_float(pari_item.get("mise", 0))
        for pari_item in paris_regles
    ]
    profits_array = np.array(profits_nets)
    if len(profits_array) == 0:
        return None, None

    succes = 0
    for _ in range(iterations):
        chantillon_simule = np.random.choice(profits_array, size=len(profits_array), replace=True)
        if np.sum(chantillon_simule) >= profit_reel:
            succes += 1

    return succes / iterations, profit_reel


def verifier_resultats_automatiques_pmu(historique):
    """
    Vérification des résultats PMU sur les paris 'En attente'.
    Garde la logique existante, mais sécurise les appels et les cas extrêmes.
    """
    modifie = False

    paris_en_attente = [
        pari_item
        for pari_item in historique
        if str(pari_item.get("statut", "")).strip().lower() == "en attente"
    ]

    if not paris_en_attente:
        return False

    groupes_paris = {}
    for pari_item in paris_en_attente:
        date_pari = str(pari_item.get("date", "")).strip()[:10]
        reunion_raw = str(pari_item.get("reunion", "")).strip()
        course_raw = str(pari_item.get("course_num", pari_item.get("course", ""))).strip()

        try:
            dt = datetime.datetime.strptime(date_pari, "%Y-%m-%d")
            date_pmu = dt.strftime("%d%m%Y")
        except Exception:
            continue

        r_match = re.search(r"\b(R\d+)\b", reunion_raw, re.IGNORECASE)
        if not r_match:
            r_match = re.search(r"R\s*(\d+)", reunion_raw, re.IGNORECASE)
            reunion_str = f"R{r_match.group(1)}" if r_match else ""
        else:
            reunion_str = r_match.group(1).upper()

        c_match = re.search(r"\b(C\d+)\b", course_raw, re.IGNORECASE)
        if not c_match:
            c_match = re.search(r"C\s*(\d+)", course_raw, re.IGNORECASE)
            course_str = f"C{c_match.group(1)}" if c_match else ""
        else:
            course_str = c_match.group(1).upper()

        if not reunion_str or not course_str:
            continue

        cle = (date_pmu, reunion_str, course_str)
        if cle not in groupes_paris:
            groupes_paris[cle] = []
        groupes_paris[cle].append(pari_item)

    total_a_verifier = sum(len(v) for v in groupes_paris.values())
    if total_a_verifier == 0:
        return False

    progress_bar = st.progress(0, text="Vérification des résultats PMU...")
    nb_traites = 0

    for cle, paris_groupe in groupes_paris.items():
        date_pmu, reunion_str, course_str = cle
        try:
            url_part = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}/{reunion_str}/{course_str}/participants"
            res_part = requests.get(url_part, headers=HEADERS, timeout=10)
            if res_part.status_code != 200:
                continue

            payload = res_part.json()
            liste_partants_bruts = payload.get("participants", []) if isinstance(payload, dict) else []

            cotes_reelles, partants_arrives = {}, []
            for part_elem in liste_partants_bruts:
                num_pmu = str(part_elem.get("numPmu"))
                rapport = part_elem.get("dernierRapportDirect")
                if isinstance(rapport, dict) and isinstance(rapport.get("rapport"), (int, float)):
                    cotes_reelles[num_pmu] = float(rapport.get("rapport"))

                ordre = part_elem.get("ordreArrivee")
                if isinstance(ordre, int) and ordre > 0:
                    partants_arrives.append((ordre, num_pmu))

            partants_arrives.sort(key=lambda x: x[0])
            arrivee_trouvee = [num for _, num in partants_arrives]

            if not arrivee_trouvee:
                continue

            partants_actifs = [
                p for p in liste_partants_bruts
                if str(p.get("traductionStatut", "")).upper() != "NON_PARTANT"
                and str(p.get("statut", "")).upper() != "NON_PARTANT"
                and str(p.get("ordreArrivee", "")).upper() != "NON_PARTANT"
            ]
            limite_places = 3 if len(partants_actifs) >= 8 else 2

            for pari_item in paris_groupe:
                nb_traites += 1
                progress_bar.progress(
                    min(1.0, nb_traites / max(total_a_verifier, 1)),
                    text=f"Vérification pari {nb_traites}/{total_a_verifier} ({pari_item.get('date', '')})...",
                )

                details = str(pari_item.get("details", pari_item.get("cheval", "")))
                mise_totale = safe_float(pari_item.get("mise", 0))
                gain_total, un_gagne = 0.0, False
                parts = details.split("|") if "|" in details else [details]

                for p_ligne in parts:
                    p_lower = p_ligne.lower()
                    nums_part = re.findall(r"N°\s*(\d+)", p_ligne)
                    mise_part_m = re.search(r"\((\d+(?:[\.,]\d+)?)\s*€\)", p_ligne)
                    mise_part = (
                        float(mise_part_m.group(1).replace(",", ".")) if mise_part_m
                        else (mise_totale / len(parts) if parts else mise_totale)
                    )

                    is_place = any(k in p_lower for k in ["placé", "place", "sécu", "ev placé"])
                    is_gagnant = any(k in p_lower for k in ["gagnant", "poker", "simple", "ev:"]) or not is_place

                    if is_place and nums_part:
                        num_secu = str(nums_part[0])

                        cote_place_reel = None
                        for p_elem in liste_partants_bruts:
                            if str(p_elem.get("numPmu")) == num_secu:
                                cote_place_reel = safe_float(p_elem.get("cote_place_reelle"), 0.0)
                                break

                        if cote_place_reel and cote_place_reel > 1.0:
                            div_ref = cote_place_reel
                        else:
                            cote_pari_origine = safe_float(pari_item.get("cote"), 3.0)
                            cote_direct = cotes_reelles.get(num_secu, cote_pari_origine)
                            div_ref = max(1.05, 1.0 + (cote_direct - 1.0) / 5.0)

                        if num_secu in arrivee_trouvee[:limite_places]:
                            gain_total += mise_part * div_ref
                            un_gagne = True

                    elif is_gagnant and nums_part:
                        num_poker = str(nums_part[0])
                        cote_pari_origine = safe_float(pari_item.get("cote"), 3.0)
                        div_ref = cotes_reelles.get(num_poker, cote_pari_origine)
                        if arrivee_trouvee and num_poker == arrivee_trouvee[0]:
                            gain_total += mise_part * div_ref
                            un_gagne = True

                pari_item["statut"] = "Gagné" if un_gagne else "Perdu"
                pari_item["gain"] = round(gain_total, 2)
                pari_item["diagnostic"] = retroaction_apprentissage_ia(
                    pari_item,
                    arrivee_trouvee,
                    cotes_reelles,
                    liste_partants_bruts,
                )

                if pari_item.get("id"):
                    update_bet_status(pari_item.get("id"), pari_item["statut"], pari_item["gain"])

                modifie = True

        except Exception as e:
            logger.error(f"Erreur vérification résultats pour la course {reunion_str}/{course_str} : {e}")
            continue

    progress_bar.empty()

    if modifie:
        sauvegarder_historique(historique)

        try:
            df_hist_suivi = load_bets_from_db()
            if not df_hist_suivi.empty:
                df_regles_suivi = df_hist_suivi[df_hist_suivi["statut"].isin(["Gagné", "Perdu"])].copy()
                if not df_regles_suivi.empty:
                    df_regles_suivi["mise"] = pd.to_numeric(df_regles_suivi["mise"], errors="coerce").fillna(0.0)
                    df_regles_suivi["gain"] = pd.to_numeric(df_regles_suivi["gain"], errors="coerce").fillna(0.0)

                    total_mises_regles = df_regles_suivi["mise"].sum()
                    total_gains_regles = df_regles_suivi[df_regles_suivi["statut"] == "Gagné"]["gain"].sum()
                    benefice_net_global = total_gains_regles - total_mises_regles

                    nouveau_montant_bankroll = 1000.0 + benefice_net_global
                    update_bankroll(nouveau_montant_bankroll)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour automatique du bankroll : {e}")

        try:
            optimiser_poids_ia_automatique()
            logger.info("Optimisation automatique des poids IA exécutée.")
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation automatique : {e}")

    return modifie


# ============================================================
# FONCTIONS DE CONTRÔLE ET FILTRE DES CANDIDATS
# ============================================================
def filtrer_candidats_value(chevaux, discipline="", min_cote=2.2, max_cote=8.5):
    """
    Filtre les chevaux sur :
    - cote dans le bon intervalle
    - EV gagnant ou placé au-dessus des seuils
    - pas de trop de candidats trop faibles
    """
    if not chevaux:
        return []

    liste = []
    for c in chevaux:
        if not isinstance(c, dict):
            continue

        cote = safe_float(c.get("cote"), 0.0)
        ev_g = safe_float(c.get("ev_index", 0.0), 0.0)
        ev_p = safe_float(c.get("ev_place_index", 0.0), 0.0)

        if not (min_cote <= cote <= max_cote):
            continue

        modele = cached_charger_modele_ia()
        disc_cfg = modele.get("disciplines_specifiques", {}).get(discipline, {})
        seuil_ev_g = safe_float(disc_cfg.get("seuil_ev_gagnant", modele.get("seuil_value_gagnant", 1.10)), 1.10)
        seuil_ev_p = safe_float(disc_cfg.get("seuil_ev_place", modele.get("seuil_value_place", 1.20)), 1.20)

        if ev_g >= seuil_ev_g or ev_p >= seuil_ev_p:
            liste.append(c)

    return sorted(
        liste,
        key=lambda c: (
            safe_float(c.get("ev_place_index", 0.0), 0.0),
            safe_float(c.get("ev_index", 0.0), 0.0),
        ),
        reverse=True,
    )


# ============================================================
# NAVIGATION PAR ONGLETS
# ============================================================
tab_accueil, tab_chrono, tab_analyse, tab_ia, tab_suivi, tab_bilan, tab_admin = st.tabs(
    [
        "🏠 Accueil",
        "⏰ Chrono des Courses",
        "📊 Analyse & Value Bets",
        "🧠 Moteur IA",
        "📈 Suivi & ROI",
        "🏟️ Bilan Réunion",
        "🛠️ Administration",
    ]
)


# ================= 1. ACCUEIL =================
with tab_accueil:
    st.title("🐎 Bienvenue sur Bilan Réunion Pro + IA")
    st.markdown("Tableau de bord centralisé pour l'analyse des courses hippiques et l'optimisation des paris.")

    bankroll_actuelle = get_bankroll()
    st.info(f"💰 **Bankroll Actuelle :** `{bankroll_actuelle:.2f} €`")

    historique = cached_charger_historique()
    total_paris = len(historique)
    gagnes = sum(1 for pari_item in historique if str(pari_item.get("statut", "")).strip().lower() == "gagné")
    mises_totales = sum(safe_float(pari_item.get("mise", 0)) for pari_item in historique)
    gains_totaux = sum(
        safe_float(pari_item.get("gain", 0))
        for pari_item in historique
        if str(pari_item.get("statut", "")).strip().lower() == "gagné"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Paris Enregistrés", total_paris)
    col2.metric("Paris Gagnants", gagnes)
    col3.metric("Mises Totales", f"{mises_totales:.2f} €")
    col4.metric("Bilan Net", f"{gains_totaux - mises_totales:.2f} €", delta=f"{gains_totaux - mises_totales:.2f} €")

    st.divider()

    st.subheader("🧠 Évolution de la Performance de l'IA")

    paris_regles = [
        p
        for p in historique
        if str(p.get("statut", "")).strip().lower() in ["gagné", "perdu"]
        and safe_float(p.get("mise"), 0.0) > 0
    ]

    if len(paris_regles) >= 5:
        df_ia = pd.DataFrame(paris_regles)
        df_ia["date"] = pd.to_datetime(df_ia["date"], errors="coerce")
        df_ia = df_ia.dropna(subset=["date"]).sort_values("date")

        df_ia["est_gagne"] = df_ia["statut"].apply(lambda s: 1 if str(s).strip().lower() == "gagné" else 0)
        df_ia["Taux de Réussite Cumulé (%)"] = (df_ia["est_gagne"].expanding().mean() * 100).round(1)

        st.markdown("##### 📈 Taux de Réussite Cumulé (%)")
        st.line_chart(df_ia.set_index("date")["Taux de Réussite Cumulé (%)"], width="stretch")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            df_ia["mise_num"] = df_ia["mise"].apply(safe_float)
            df_ia["gain_num"] = df_ia["gain"].apply(safe_float)
            df_ia["profit_cumul"] = (df_ia["gain_num"] - df_ia["mise_num"]).cumsum()
            df_ia["mises_cumul"] = df_ia["mise_num"].cumsum()
            df_ia["ROI Cumulé (%)"] = ((df_ia["profit_cumul"] / df_ia["mises_cumul"]) * 100).fillna(0).round(1)

            st.markdown("##### 📊 ROI Cumulé (%)")
            st.line_chart(df_ia.set_index("date")["ROI Cumulé (%)"], width="stretch")

        with col_p2:
            st.info(
                """
                **Lecture rapide :**
                * **Taux de Réussite Cumulé** : montre la qualité globale de ciblage.
                * **ROI Cumulé** : montre si la stratégie est rentable sur le long terme.
                """
            )
    else:
        st.info("Il faut au moins 5 paris réglés pour afficher la courbe de progression de l'IA.")


# ================= 2. CHRONO DES COURSES =================
with tab_chrono:
    st.subheader("⏰ Programme Chronologique & Sélection Value Bets")
    col_c1, col_c2 = st.columns([2, 2])
    with col_c1:
        date_chrono_sel = st.date_input("Date", key="date_chrono_picker")
        date_chrono_iso = date_chrono_sel.strftime("%Y-%m-%d")

    with col_c2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            btn_charger = st.button("🔍 Afficher les courses", key="btn_afficher_chrono")
        with col_btn2:
            btn_telecharger = st.button("📥 Télécharger/Actualiser", key="btn_dl_chrono")

    if btn_telecharger:
        with st.spinner("Téléchargement des programmes en cours..."):
            if telecharger_pmu_date(date_chrono_iso, None):
                st.success("Données actualisées et stockées en base !")
                donnees_chrono, _ = cached_charger_courses_jour_db(date_chrono_iso)
                st.session_state["cache_chrono_donnees"] = donnees_chrono
                st.session_state["cache_chrono_date"] = date_chrono_iso
            else:
                st.warning("Aucune donnée récupérée pour cette date.")

    if btn_charger or st.session_state.get("cache_chrono_date") == date_chrono_iso:
        if "cache_chrono_donnees" not in st.session_state or st.session_state.get("cache_chrono_date") != date_chrono_iso:
            with st.spinner("Chargement des courses depuis la base..."):
                donnees_chrono, _ = cached_charger_courses_jour_db(date_chrono_iso)
                st.session_state["cache_chrono_donnees"] = donnees_chrono
                st.session_state["cache_chrono_date"] = date_chrono_iso
        else:
            donnees_chrono = st.session_state["cache_chrono_donnees"]

        if donnees_chrono:
            toutes_courses = []

            for c_elem in donnees_chrono:
                if not isinstance(c_elem, dict):
                    continue

                nom_c = str(c_elem.get("nom_course", "")).strip()
                if not nom_c or nom_c.isdigit() or len(nom_c) <= 2:
                    continue

                r_nom = f"{c_elem.get('reunion', 'R1')} - {c_elem.get('hippodrome', 'HIPPODROME')}"
                chevaux_c = c_elem.get("chevaux", [])

                if not chevaux_c:
                    continue

                modele = cached_charger_modele_ia()
                disc_cfg = modele.get("disciplines_specifiques", {}).get(c_elem.get("discipline", ""), {})
                min_cote = safe_float(disc_cfg.get("min_cote", 2.20), 2.20)
                max_cote = safe_float(disc_cfg.get("max_cote", 8.50), 8.50)

                chevaux_val_c = [c for c in chevaux_c if isinstance(c, dict) and min_cote <= safe_float(c.get("cote"), 0.0) <= max_cote]

                base_chev = {"num": "?", "nom": "Inconnu", "cote": 0.0, "ev_index": 0.0, "ev_place_index": 0.0}
                poker_chev = {"num": "?", "nom": "Inconnu", "cote": 0.0, "ev_index": 0.0, "ev_place_index": 0.0}

                if chevaux_val_c:
                    params_ad_chrono = cached_calculer_parametres_adaptatifs()
                    for c in chevaux_val_c:
                        c["score_analyse"] = evaluer_score_cheval(
                            c,
                            c_elem.get("discipline"),
                            c_elem.get("terrain_officiel"),
                            c_elem.get("corde", "Corde standard"),
                            date_chrono_iso,
                            params_ad_chrono,
                            distance_course=c_elem.get("distance", ""),
                            hippodrome=c_elem.get("hippodrome", ""),
                            utiliser_ml=True,
                        )

                    normaliser_scores_chevaux(chevaux_val_c, "score_analyse")
                    calculer_valeur_esperee_avancee(chevaux_val_c, len(chevaux_c))

                    chevaux_val_c.sort(
                        key=lambda x: (
                            safe_float(x.get("ev_place_index", 0)),
                            safe_float(x.get("ev_index", 0)),
                        ),
                        reverse=True,
                    )

                    seuil_ev_g = safe_float(disc_cfg.get("seuil_ev_gagnant", 1.10), 1.10)
                    seuil_ev_p = safe_float(disc_cfg.get("seuil_ev_place", 1.20), 1.20)

                    candidats_valides = [
                        c
                        for c in chevaux_val_c
                        if safe_float(c.get("ev_index", 0)) >= seuil_ev_g
                        or safe_float(c.get("ev_place_index", 0)) >= seuil_ev_p
                    ]

                    if candidats_valides:
                        base_chev = candidats_valides[0]
                        outsiders_c = [
                            c
                            for c in candidats_valides
                            if 7.0 <= safe_float(c.get("cote")) <= 12.0 and str(c.get("num")) != str(base_chev.get("num"))
                        ]
                        poker_chev = (
                            max(outsiders_c, key=lambda x: safe_float(x.get("ev_index", 0)))
                            if outsiders_c
                            else (candidats_valides[1] if len(candidats_valides) > 1 else base_chev)
                        )

                        toutes_courses.append(
                            {
                                "heure": c_elem.get("heure", "13:30"),
                                "reunion": r_nom,
                                "course_num": c_elem.get("course", "C1"),
                                "nom_course": nom_c,
                                "discipline": c_elem.get("discipline", ""),
                                "data": c_elem,
                                "base": base_chev,
                                "poker": poker_chev,
                            }
                        )

            toutes_courses.sort(key=lambda x: x["heure"])

            st.success(f"{len(toutes_courses)} course(s) chargée(s) avec des opportunités valides pour le {date_chrono_iso}.")

            for idx_c, item_c in enumerate(toutes_courses):
                course_obj = item_c["data"]
                b_chev = item_c["base"]
                p_chev = item_c["poker"]
                cle_unique_course = f"{item_c['reunion']}_{item_c['course_num']}_{idx_c}"

                with st.expander(
                    f"🕒 {item_c['heure']} | {item_c['reunion']} ➔ {item_c['course_num']} : {item_c['nom_course']}"
                ):
                    st.markdown(
                        f"**Base Value Bet (Sécurité) :** Simple Placé ➔ N°{b_chev.get('num')} - {b_chev.get('nom')} "
                        f"(Cote Placé: {safe_float(b_chev.get('cote_place_estimee', 0), 0):.1f} | "
                        f"**EV Placé: {safe_float(b_chev.get('ev_place_index', 0), 0):.2f}**) "
                    )
                    st.markdown(
                        f"**Coup de Poker Value :** Simple Gagnant ➔ N°{p_chev.get('num')} - {p_chev.get('nom')} "
                        f"(Cote: {safe_float(p_chev.get('cote')):.1f} | **EV: {safe_float(p_chev.get('ev_index', 0), 0):.2f}**) "
                    )

                    col_m, col_b = st.columns([2, 1])
                    with col_m:
                        mise_input = st.number_input("Mise Totale (€)", min_value=1, value=10, key=f"m_{cle_unique_course}")
                        mise_secu = round(mise_input * 0.75, 1)
                        mise_poker = round(mise_input - mise_secu, 1)
                        st.caption(
                            f"💡 Répartition Sécurisée : **{mise_secu} €** Sécu (75%) | **{mise_poker} €** Poker (25%)"
                        )

                    with col_b:
                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                        if st.button("⚡ Valider & Enregistrer", key=f"btn_valider_{cle_unique_course}"):
                            enregistrer_pari_via_ia(
                                date_iso=date_chrono_iso,
                                reunion=item_c["reunion"],
                                course_num=item_c["course_num"],
                                discipline=course_obj.get("discipline"),
                                cheval_data=b_chev,
                                type_pari_souhaite="Simple Placé (Sécurité Value)",
                                mise_totale=mise_secu,
                            )
                            if mise_poker > 0:
                                enregistrer_pari_via_ia(
                                    date_iso=date_chrono_iso,
                                    reunion=item_c["reunion"],
                                    course_num=item_c["course_num"],
                                    discipline=course_obj.get("discipline"),
                                    cheval_data=p_chev,
                                    type_pari_souhaite="Simple Gagnant (Poker Value)",
                                    mise_totale=mise_poker,
                                )
                            st.success("Paris validés, analysés et enregistrés via l'IA !")
                            st.rerun()
        else:
            st.info("Aucune donnée enregistrée en base pour cette date. Cliquez sur 'Télécharger/Actualiser'.")
    else:
        st.info("Sélectionnez une date puis cliquez sur **'Afficher les courses'** pour charger le programme de la journée.")


# ================= 3. ANALYSE & VALUE BETS =================
with tab_analyse:
    st.subheader("📊 Analyse Intégrale & Détection de Value Bets")
    date_sel = st.date_input("Date du jour", key="date_analyse_picker")
    date_iso = date_sel.strftime("%Y-%m-%d")

    donnees, reunions_map = cached_charger_courses_jour_db(date_iso)
    if reunions_map:
        reunion_choisie = st.selectbox("Réunion", sorted(list(reunions_map.keys())))
        courses = reunions_map[reunion_choisie]
        c_map = {f"{c['course']} : {c['nom_course']}": c for c in courses}
        c_choisie = st.selectbox("Course", list(c_map.keys()))
        course_curr = c_map[c_choisie]

        st.info(f"Terrain : {course_curr.get('terrain_officiel')} | Discipline : {course_curr.get('discipline')}")

        st.divider()
        st.markdown("### ⚙️ Configuration du Pari (Avant Analyse)")
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            type_pari_sel = st.selectbox(
                "Type de pari",
                ["Simple Gagnant", "Simple Placé", "Value Bet IA"],
                key="sel_type_pari_analyse",
            )
        with col_cfg2:
            mise_analyse = st.number_input("Mise (€)", min_value=1.0, value=10.0, step=1.0, key="mise_analyse_input")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        if st.button("⚡ Lancer l'Analyse Maximisation Gains IA", key="btn_lancer_analyse_ia"):
            params = cached_calculer_parametres_adaptatifs()
            for c in course_curr.get("chevaux", []):
                if isinstance(c, dict):
                    c["score_ia"] = evaluer_score_cheval(
                        c,
                        course_curr.get("discipline"),
                        course_curr.get("terrain_officiel"),
                        course_curr.get("corde", "Corde standard"),
                        date_iso,
                        params,
                        distance_course=course_curr.get("distance", ""),
                        hippodrome=course_curr.get("hippodrome", ""),
                        utiliser_ml=True,
                    )

            normaliser_scores_chevaux(course_curr.get("chevaux", []), "score_ia")
            calculer_valeur_esperee_avancee(course_curr.get("chevaux", []), len(course_curr.get("chevaux", [])))

            chevaux_tries = sorted(course_curr.get("chevaux", []), key=lambda x: x.get("ev_index", 0) if isinstance(x, dict) else 0, reverse=True)
            st.session_state["chevaux_analyse_courant"] = chevaux_tries

            st.dataframe(
                [
                    {
                        "N°": c["num"],
                        "Nom": c["nom"],
                        "Driver": c.get("driver"),
                        "Cote": c.get("cote"),
                        "Score IA": c.get("score_ia"),
                        "Proba (Softmax)": f"{safe_float(c.get('proba_estimee', 0)) * 100:.1f}%",
                        "EV Gagnant": c.get("ev_index"),
                        "EV Placé": c.get("ev_place_index"),
                    }
                    for c in chevaux_tries
                    if isinstance(c, dict)
                ],
                width="stretch",
            )

        if "chevaux_analyse_courant" in st.session_state and st.session_state["chevaux_analyse_courant"]:
            st.divider()
            st.markdown("### 📝 Enregistrer un pari sur cette course analysée")
            chevaux_analyse = st.session_state["chevaux_analyse_courant"]

            options_chevaux = [
                f"N°{c['num']} - {c['nom']} (Cote: {safe_float(c.get('cote')):.1f})"
                for c in chevaux_analyse
                if isinstance(c, dict)
            ]
            cheval_selectionne_str = st.selectbox("Sélectionner le cheval", options_chevaux, key="sel_cheval_analyse")

            st.info(f"Configuration active : **{type_pari_sel}** | Mise : **{mise_analyse} €**")

            if st.button("⚡ Valider & Enregistrer ce Pari", key="btn_valider_analyse_cours"):
                match_num = re.search(r"N°(\d+)", cheval_selectionne_str)
                num_cheval = match_num.group(1) if match_num else "?"
                cheval_obj = next(
                    (c for c in chevaux_analyse if isinstance(c, dict) and str(c.get("num")) == str(num_cheval)),
                    chevaux_analyse[0] if chevaux_analyse else {},
                )

                enregistrer_pari_via_ia(
                    date_iso=date_iso,
                    reunion=reunion_choisie,
                    course_num=course_curr.get("course"),
                    discipline=course_curr.get("discipline"),
                    cheval_data=cheval_obj,
                    type_pari_souhaite=type_pari_sel,
                    mise_totale=mise_analyse,
                )
                st.success("Pari validé et enregistré avec succès via l'IA depuis l'onglet Analyse !")
                st.rerun()

        st.divider()
        st.subheader("💰 Allocation Stratégique (Critère de Kelly)")

        budget_saisi = st.number_input("Budget Global à Allouer (€)", min_value=5, max_value=500, value=50, step=5, key="budget_kelly")

        if st.button("🎲 Calculer le Plan d'Allocation Optimal", key="btn_calc_kelly"):
            params_ad = cached_calculer_parametres_adaptatifs()
            plan = generer_plan_budget_journalier(date_iso, budget_saisi, params_ad)
            if plan:
                st.session_state["plan_courant"] = plan
            else:
                st.session_state["plan_courant"] = None
                st.info("Aucune opportunité ne présente un EV >= 1.50 aujourd'hui.")

        if st.session_state.get("plan_courant"):
            st.write("### 📌 Stratégie de Mises Optimisée")
            plan_data = st.session_state["plan_courant"]
            if isinstance(plan_data, pd.DataFrame):
                st.dataframe(plan_data, width="stretch")
                plan_df = plan_data
            else:
                plan_df = pd.DataFrame(plan_data)
                st.dataframe(plan_df, width="stretch")

            if st.button("⚡ Valider & Enregistrer le Plan d'Allocation Optimal", key="btn_valider_plan_allocation"):
                paris_a_inserer = []

                for _, row in plan_df.iterrows():
                    reunion = str(row.get("Réunion", row.get("reunion", "R1")))
                    course_num = str(row.get("Course", row.get("course_num", row.get("course", "C1"))))
                    cheval_nom = str(row.get("Cheval", row.get("cheval", row.get("nom", ""))))
                    num_cheval = str(row.get("N°", row.get("num", "")))
                    mise = safe_float(row.get("Mise (€)", row.get("mise", 10.0)))
                    type_pari = str(row.get("Type", row.get("type_pari", "Allocation Kelly")))
                    cote = safe_float(row.get("Cote", 0.0))

                    details_str = (
                        f"{type_pari} ➔ N°{num_cheval} {cheval_nom} "
                        f"(Cote: {cote} | EV: {max(safe_float(row.get('EV', 0)), 0):.2f})"
                    )

                    paris_a_inserer.append(
                        {
                            "date": date_iso,
                            "reunion": reunion,
                            "course": course_num,
                            "cheval": details_str,
                            "cote": cote,
                            "mise": mise,
                            "statut": "En attente",
                            "gain": 0.0,
                            "discipline": course_curr.get("discipline", "Galop Plat"),
                        }
                    )

                if paris_a_inserer:
                    succes = save_multiple_bets_to_db(paris_a_inserer)
                    if succes:
                        st.success(f"✅ {len(paris_a_inserer)} paris du plan d'allocation optimal ont été validés et enregistrés instantanément !")
                        st.session_state.pop("plan_courant", None)
                        st.rerun()
                else:
                    st.warning("Aucun pari trouvé dans le plan d'allocation.")


# ================= 4. MOTEUR IA & PARAMÈTRES =================
with tab_ia:
    st.subheader("🧠 Config & Poids du Moteur IA par Discipline")
    modele = cached_charger_modele_ia()

    if "disciplines_specifiques" not in modele:
        modele["disciplines_specifiques"] = {}

    disciplines_disponibles = ["Trot Attelé", "Galop Plat", "Haies", "Steeple-chase", "Trot Monté"]
    discipline_selectionnee = st.selectbox(
        "🎯 Sélectionner la discipline à configurer",
        disciplines_disponibles,
        key="select_discipline_ia",
    )

    disc_dict = modele["disciplines_specifiques"].get(discipline_selectionnee, {})

    col_ia1, col_ia2 = st.columns(2)
    with col_ia1:
        st.subheader(f"⚙️ Poids Heuristiques : {discipline_selectionnee}")
        with st.form(f"form_poids_ia_{discipline_selectionnee}"):
            poids_musique = st.slider(
                "Poids Musique",
                0.1,
                10.0,
                float(disc_dict.get("poids_musique", modele.get("poids_musique", 1.05))),
                0.05,
            )
            poids_ferrage = st.slider(
                "Poids Ferrage",
                0.1,
                10.0,
                float(disc_dict.get("poids_ferrage", modele.get("poids_ferrage", 1.25))),
                0.05,
            )
            poids_terrain = st.slider(
                "Poids Terrain",
                0.1,
                10.0,
                float(disc_dict.get("poids_terrain", modele.get("poids_terrain", 1.15))),
                0.05,
            )
            poids_poids = st.slider(
                "Poids Poids",
                0.1,
                10.0,
                float(disc_dict.get("poids_poids", modele.get("poids_poids", 1.0))),
                0.05,
            )
            poids_cote_tendance = st.slider(
                "Poids Tendance Cote",
                0.1,
                10.0,
                float(disc_dict.get("poids_cote_tendance", modele.get("poids_cote_tendance", 1.35))),
                0.05,
            )
            poids_driver = st.slider(
                "Poids Driver",
                0.1,
                10.0,
                float(disc_dict.get("poids_driver", modele.get("poids_driver", 1.15))),
                0.05,
            )
            poids_corde = st.slider(
                "Poids Corde",
                0.1,
                10.0,
                float(disc_dict.get("poids_corde", modele.get("poids_corde", 1.0))),
                0.05,
            )
            poids_hippodrome = st.slider(
                "Poids Hippodrome/Acteur",
                0.1,
                10.0,
                float(disc_dict.get("poids_hippodrome_acteur", modele.get("poids_hippodrome_acteur", 1.25))),
                0.05,
            )
            poids_distance = st.slider(
                "Poids Distance",
                0.1,
                10.0,
                float(disc_dict.get("poids_distance", modele.get("poids_distance", 1.10))),
                0.05,
            )
            poids_outsider_cache = st.slider(
                "Poids Outsiders Cachés",
                0.1,
                10.0,
                float(disc_dict.get("poids_outsider_cache", modele.get("poids_outsider_cache", 1.30))),
                0.05,
            )

            st.markdown("---")
            st.markdown("##### 🎯 Filtres de Cotes & Seuils EV par Discipline")
            min_cote_input = st.number_input("Cote Minimale Filtrée", 1.0, 50.0, float(disc_dict.get("min_cote", 2.20)), 0.10)
            max_cote_input = st.number_input("Cote Maximale Filtrée", 1.0, 50.0, float(disc_dict.get("max_cote", 8.50)), 0.10)

            seuil_ev_gagnant_input = st.number_input("Seuil EV Gagnant", 1.0, 5.0, float(disc_dict.get("seuil_ev_gagnant", 1.10)), 0.05)
            seuil_ev_place_input = st.number_input("Seuil EV Placé", 1.0, 5.0, float(disc_dict.get("seuil_ev_place", 1.20)), 0.05)

            seuil_ev = st.number_input("Seuil Minimal Value Bet Global (EV)", 1.0, 10.0, float(modele.get("seuil_value_bet", 1.50)), 0.05)
            bloque_disc = st.checkbox(
                "🛑 Bloquer les paris sur cette discipline (Quarantaine)",
                value=disc_dict.get("bloque_paris", False),
            )

            if st.form_submit_button(f"💾 Sauvegarder pour {discipline_selectionnee}"):
                modele["disciplines_specifiques"][discipline_selectionnee] = {
                    "poids_musique": poids_musique,
                    "poids_ferrage": poids_ferrage,
                    "poids_terrain": poids_terrain,
                    "poids_poids": poids_poids,
                    "poids_cote_tendance": poids_cote_tendance,
                    "poids_driver": poids_driver,
                    "poids_corde": poids_corde,
                    "poids_hippodrome_acteur": poids_hippodrome,
                    "poids_distance": poids_distance,
                    "poids_outsider_cache": poids_outsider_cache,
                    "min_cote": min_cote_input,
                    "max_cote": max_cote_input,
                    "seuil_ev_gagnant": seuil_ev_gagnant_input,
                    "seuil_ev_place": seuil_ev_place_input,
                    "bloque_paris": bloque_disc,
                }
                modele["seuil_value_bet"] = seuil_ev

                sauvegarder_modele_ia(modele)
                st.success(f"Modèle mis à jour avec succès pour le {discipline_selectionnee} !")
                st.rerun()

    with col_ia2:
        st.subheader(f"📊 Statistiques d'Impact : {discipline_selectionnee}")

        df_all_bets = cached_load_bets_from_db()
        if not df_all_bets.empty and "discipline" in df_all_bets.columns:
            df_disc = df_all_bets[df_all_bets["discipline"] == discipline_selectionnee]
            nb_analyses_disc = len(df_disc)

            df_regle_disc = df_disc[df_disc["statut"].isin(["Gagné", "Perdu"])]
            mises_disc = pd.to_numeric(df_regle_disc["mise"], errors="coerce").sum()
            gains_disc = pd.to_numeric(df_regle_disc["gain"], errors="coerce").sum()
            gain_net_disc = gains_disc - mises_disc
        else:
            nb_analyses_disc = 0
            gain_net_disc = 0.0

        st.metric(f"Total Paris / Analyses ({discipline_selectionnee})", nb_analyses_disc)
        st.metric(f"Bilan Net IA ({discipline_selectionnee})", f"{gain_net_disc:+.2f} €", delta=f"{gain_net_disc:+.2f} €")

        stats = modele.get("stats_impact", {})
        st.metric("Victoires globales liées au Ferrage", stats.get("victoires_par_ferrage", 0))

        st.divider()
        st.subheader("🧬 Apprentissage & Optimisation")
        st.markdown("Lance l'optimisation des poids heuristiques basée sur l'historique des courses enregistrées.")

        if st.button("🚀 Lancer l'optimisation automatique des poids", key="btn_lancer_optim_ia"):
            succes, msg = optimiser_poids_ia_automatique()
            if succes:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

        params_actuels = cached_calculer_parametres_adaptatifs()
        st.info(f"💡 Message Modèle Adaptatif : {params_actuels.get('message_auto')}")


# ================= 5. SUIVI & ROI FINANCIER =================
with tab_suivi:
    st.subheader("📈 Suivi Financier, ROI & Progression Course par Course")
    if st.button("🔄 Vérifier Automatiquement les Résultats PMU", key="btn_verif_pmu_auto"):
        hist = cached_charger_historique()
        if verifier_resultats_automatiques_pmu(hist):
            sauvegarder_historique(hist)
            st.success("Résultats PMU mis à jour !")
            st.rerun()
        else:
            st.info("Aucun résultat supplémentaire trouvé.")

    historique = cached_charger_historique()
    paris_regles = [
        pari_item
        for pari_item in historique
        if str(pari_item.get("statut", "")).strip().lower() in ["gagné", "perdu", "remboursé"]
    ]

    total_mises = sum(safe_float(pari_item.get("mise", 0)) for pari_item in paris_regles)
    total_gains = sum(
        safe_float(pari_item.get("gain", 0))
        for pari_item in paris_regles
        if str(pari_item.get("statut", "")).strip().lower() in ["gagné", "remboursé"]
    )
    bilan_net = total_gains - total_mises
    roi = (bilan_net / total_mises * 100) if total_mises > 0 else 0.0
    paris_valides_win = [p for p in paris_regles if str(p.get("statut", "")).strip().lower() in ["gagné", "perdu"]]
    taux_win = (
        sum(1 for pari_item in paris_valides_win if str(pari_item.get("statut", "")).strip().lower() == "gagné")
        / len(paris_valides_win) * 100
    ) if paris_valides_win else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Mises", f"{total_mises:.2f} €")
    kpi2.metric("Total Gains", f"{total_gains:.2f} €")
    kpi3.metric("Bilan Net", f"{bilan_net:+.2f} €", delta=f"{bilan_net:+.2f} €")
    kpi4.metric("ROI / Taux Réussite", f"{roi:+.1f}% | Win: {taux_win:.1f}%")

    st.divider()
    st.subheader("🎯 Évolution Chronologique (Pari par Pari / Course par Course)")

    if len(paris_regles) >= 2:
        df_chrono = pd.DataFrame(paris_regles)
        df_chrono["date"] = pd.to_datetime(df_chrono["date"], errors="coerce")
        df_chrono = df_chrono.dropna(subset=["date"]).sort_values(by=["date", "reunion", "course"]).reset_index(drop=True)

        def calc_benefice_net(row):
            statut = str(row.get("statut", "")).strip().lower()
            mise = safe_float(row.get("mise", 0))
            gain = safe_float(row.get("gain", 0))
            if statut == "gagné":
                return gain - mise
            elif statut == "remboursé":
                return 0.0
            else:
                return -mise

        df_chrono["benefice_net"] = df_chrono.apply(calc_benefice_net, axis=1)
        df_chrono["Mises Cumulées (€)"] = df_chrono["mise"].apply(safe_float).cumsum()
        df_chrono["Bilan Net Cumulé (€)"] = df_chrono["benefice_net"].cumsum()
        df_chrono["ROI Cumulé (%)"] = (
            (df_chrono["Bilan Net Cumulé (€)"] / df_chrono["Mises Cumulées (€)"]) * 100
        ).fillna(0).round(2)
        df_chrono["Index Course"] = [f"Pari #{i + 1}" for i in range(len(df_chrono))]

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### 📈 Courbe du Bilan Net Cumulé (€)")
            st.line_chart(df_chrono.set_index("Index Course")["Bilan Net Cumulé (€)"], width="stretch")

        with col_g2:
            st.markdown("##### 📊 Courbe du ROI Cumulé (%)")
            st.line_chart(df_chrono.set_index("Index Course")["ROI Cumulé (%)"], width="stretch")

        with st.expander("🔍 Voir le détail du tableau chronologique des paris", expanded=False):
            st.dataframe(
                df_chrono[["date", "reunion", "course", "cheval", "mise", "statut", "gain", "Bilan Net Cumulé (€)", "ROI Cumulé (%)"]],
                width="stretch",
            )
    else:
        st.info("Il faut au moins 2 paris réglés pour afficher l'évolution chronologique.")

    st.divider()
    st.subheader("🧪 Tests de Significativité Statistique")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        t_score, mean_r = test_student_yield(paris_regles)
        if t_score is not None:
            st.write("**Test t de Student :**")
            st.write(f"- Rendement Moyen : `{mean_r * 100:.2f}%` | t-score : `{t_score:.2f}`")
        else:
            st.info("Pas assez de paris pour le test t (min. 2).")

    with col_t2:
        p_val, prof_r = test_significativite_monte_carlo(paris_regles)
        if p_val is not None:
            st.write("**Simulation Monte Carlo :**")
            st.write(f"- Profit : `{prof_r:.2f} €` | p-value : `{p_val:.4f}`")
        else:
            st.info("Pas assez de paris pour Monte Carlo (min. 10).")

    st.divider()
    st.subheader("📋 Historique Détaillé des Paris (Paginé)")
    if historique:
        df_hist = pd.DataFrame(historique)
        elements_par_page = 25
        total_pages = max(1, math.ceil(len(df_hist) / elements_par_page))
        page_courante = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        debut = (page_courante - 1) * elements_par_page
        fin = debut + elements_par_page

        st.dataframe(df_hist.iloc[debut:fin], width="stretch")
        st.caption(f"Affichage des paris {debut + 1} à {min(fin, len(df_hist))} sur un total de {len(df_hist)}")
    else:
        st.info("Aucun pari enregistré.")


# ================= 6. BILAN PAR RÉUNION =================
with tab_bilan:
    st.subheader("🏟️ Bilan Financier par Réunion & Discipline")

    st.subheader("📊 Performance par Discipline")
    df_sql = analyser_roi_par_discipline_sql()
    if not df_sql.empty:
        st.dataframe(df_sql, width="stretch")
    else:
        st.info("Aucune donnée disponible pour l'analyse par discipline.")

    st.divider()
    st.subheader("📅 Bilans Journaliers")
    df_journalier = analyser_bilan_journalier_sql()
    if not df_journalier.empty:
        st.dataframe(df_journalier, width="stretch")
    else:
        st.info("Aucun pari réglé disponible pour afficher le bilan journalier.")


# ================= 7. ADMINISTRATION =================
with tab_admin:
    st.subheader("🛠️ Panneau d'Administration Avancé")

    with st.form("form_admin_login"):
        mdp_admin = st.text_input("Code Admin requis", type="password")
        submit_admin = st.form_submit_button("Valider l'accès")

    if submit_admin and mdp_admin.strip() == st.secrets.get("PASSWORD", "301180"):
        st.session_state["admin_authentifie"] = True

    if st.session_state.get("admin_authentifie", False):
        st.success("Accès administrateur autorisé.")

        st.divider()
        st.subheader("🤖 Apprentissage Séquentiel (Jour par Jour)")
        st.markdown("L'IA traite les courses **jour après jour**, vérifie les résultats et s'adapte immédiatement.")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            date_debut = st.date_input("Date de début", key="admin_date_debut")
        with col_d2:
            date_fin = st.date_input("Date de fin", key="admin_date_fin")

        mise_auto_defaut = st.number_input("Mise par course par défaut (€)", min_value=1, value=10, key="admin_mise_auto")

        if st.button("🚀 Lancer l'apprentissage séquentiel quotidien", key="btn_lancer_auto_periode"):
            if date_debut > date_fin:
                st.error("La date de début doit être antérieure ou égale à la date de fin.")
            else:
                optimiser_poids_ia_automatique()

                roi_global = -99.0
                tentative = 0
                max_tentatives = 1
                total_paris_ajoutes = 0

                st.markdown("### 📊 Suivi en direct")
                col_suivi1, col_suivi2 = st.columns(2)
                with col_suivi1:
                    box_statut = st.empty()
                with col_suivi2:
                    box_roi = st.empty()

                box_details = st.empty()
                table_suivi_container = st.empty()
                historique_journalier = []

                while roi_global < 0.0 and tentative < max_tentatives:
                    tentative += 1
                    total_paris_ajoutes = 0
                    courante_dt = date_debut
                    delta = datetime.timedelta(days=1)
                    nb_jours = (date_fin - date_debut).days + 1

                    box_statut.metric("Tentative d'optimisation globale", f"{tentative} / {max_tentatives}")

                    mises_totales_periode = 0.0
                    gains_totaux_periode = 0.0

                    for _ in range(nb_jours):
                        courante_iso = courante_dt.strftime("%Y-%m-%d")
                        box_details.info(f"📅 [Tentative {tentative}] Traitement et paris pour la journée : **{courante_iso}**")

                        gains_jour = 0.0
                        mises_jour = 0.0
                        nb_c_jour = 0

                        try:
                            try:
                                conn_net = get_conn()
                                cur_net = conn_net.cursor()
                                cur_net.execute(
                                    "DELETE FROM pmu_bets WHERE date = %s AND (cheval LIKE %s OR cheval LIKE %s OR cheval LIKE %s)",
                                    (courante_iso, "Simulation%", "Observation%", "%Value Auto%"),
                                )
                                conn_net.commit()
                            except Exception as e_net:
                                logger.error(f"Erreur lors du nettoyage préventif des paris pour la date {courante_iso} : {e_net}")
                            finally:
                                try:
                                    cur_net.close()
                                except Exception:
                                    pass
                                try:
                                    conn_net.close()
                                except Exception:
                                    pass

                            nb_c_jour, mises_jour = simuler_et_entrainer_journee_intelligente(courante_iso, float(mise_auto_defaut))
                            total_paris_ajoutes += nb_c_jour

                            box_details.info(f"🔍 [Tentative {tentative}] Vérification PMU et adaption IA pour le {courante_iso}...")
                            hist_actuel = cached_charger_historique()
                            if verifier_resultats_automatiques_pmu(hist_actuel):
                                sauvegarder_historique(hist_actuel)

                            optimiser_poids_ia_automatique()

                            hist_verifie = cached_charger_historique()
                            for p_item in hist_verifie:
                                if str(p_item.get("date", ""))[:10] == courante_iso:
                                    if p_item.get("statut") in ["Gagné", "Remboursé"]:
                                        gains_jour += safe_float(p_item.get("gain", 0.0))

                            mises_totales_periode += mises_jour
                            gains_totaux_periode += gains_jour

                            roi_cumule_periode = (
                                ((gains_totaux_periode - mises_totales_periode) / mises_totales_periode * 100)
                                if mises_totales_periode > 0
                                else 0.0
                            )

                            historique_journalier.append(
                                {
                                    "Tentative": f"Tentative {tentative}",
                                    "Date": courante_iso,
                                    "Paris ajoutés (jour)": nb_c_jour,
                                    "ROI Global Cumulé (%)": round(roi_cumule_periode, 2),
                                }
                            )
                            table_suivi_container.dataframe(pd.DataFrame(historique_journalier), width="stretch")

                        except Exception as e:
                            logger.error(f"Erreur date {courante_iso} : {e}")

                        courante_dt += delta

                    roi_global = (
                        ((gains_totaux_periode - mises_totales_periode) / mises_totales_periode * 100)
                        if mises_totales_periode > 0
                        else 0.0
                    )

                    box_roi.metric("ROI Global de la Période", f"{roi_global:+.2f}%")
                    box_details.success(f"📌 Fin de la tentative globale {tentative} - ROI obtenu : {roi_global:+.2f}%")

                    if roi_global >= 0.0:
                        break

                st.balloons()
                st.success(f"🎉 Processus séquentiel terminé en {tentative} tentative(s) ! {total_paris_ajoutes} paris traités au total.")
                st.metric("ROI Global Final Obtenu", f"{roi_global:+.2f}%")

        st.divider()
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("🧹 Vider le cache complet Streamlit", key="btn_vider_cache_admin"):
                st.cache_data.clear()
                st.success("Cache vidé !")
        with col_a2:
            if st.button("🔄 Forcer la migration Base de Données", key="btn_migrer_db_admin"):
                init_db()
                st.success("Initialisation de la base de données effectuée.")

        st.divider()
        st.subheader("🧹 Nettoyage des Paris en Attente Expirés")
        st.markdown("Supprime les paris restés 'En attente' dont la date est trop ancienne.")

        col_n1, col_n2 = st.columns([2, 1])
        with col_n1:
            nb_jours_del = st.number_input("Ancienneté minimale (en jours)", min_value=1, value=7, step=1, key="input_nb_jours_del_attente")
        with col_n2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Nettoyer les paris en attente", key="btn_nettoyer_attente_vieus"):
                try:
                    date_limite = datetime.date.today() - datetime.timedelta(days=int(nb_jours_del))
                    date_limite_str = date_limite.strftime("%Y-%m-%d")

                    conn = get_conn()
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM pmu_bets WHERE statut = %s AND date < %s",
                        ("En attente", date_limite_str),
                    )
                    nb_supprimes = cursor.rowcount
                    conn.commit()
                    cursor.close()
                    conn.close()

                    hist = cached_charger_historique()
                    nouveaux_hist = []
                    supp_hist_count = 0
                    for pari_item in hist:
                        if pari_item.get("statut") == "En attente":
                            d_str = str(pari_item.get("date", ""))[:10]
                            try:
                                if d_str and datetime.datetime.strptime(d_str, "%Y-%m-%d").date() < date_limite:
                                    supp_hist_count += 1
                                    continue
                            except Exception:
                                pass
                        nouveaux_hist.append(pari_item)

                    if supp_hist_count > 0:
                        sauvegarder_historique(nouveaux_hist)

                    total_effectif = max(nb_supprimes, supp_hist_count)
                    st.success(f"🧹 Nettoyage réussi : {total_effectif} pari(s) en attente de plus de {nb_jours_del} jours ont été supprimés.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors du nettoyage des paris en attente : {e}")

        st.divider()
        st.subheader("⚠️ Zone Dangereuse : Remise à Zéro Complète")
        st.markdown("Cette action supprimera définitivement tous les paris, le cache et les modèles.")

        confirm_raz = st.checkbox("Je confirme vouloir supprimer TOUTES les données (irréversible)", key="confirm_raz_checkbox")
        if confirm_raz:
            if st.button("🗑️ Supprimer TOUTES les données acquises", key="btn_raz_total", type="primary"):
                try:
                    conn = get_conn()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM pmu_bets;")
                    cursor.execute("DELETE FROM pmu_bankroll;")
                    cursor.execute("DELETE FROM pmu_predictions_cache;")
                    cursor.execute("DELETE FROM courses_cache;")
                    cursor.execute("DELETE FROM modele_ia;")
                    conn.commit()
                    cursor.close()
                    conn.close()

                    st.cache_data.clear()
                    st.success("🗑️ Remise à zéro effectuée avec succès !")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la remise à zéro : {e}")