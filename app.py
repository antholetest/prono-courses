# app.py
import datetime
import json
import logging
from pathlib import Path
import math
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# --- IMPORTATIONS DES MODULES INTERNES (Ajustées pour inclure analyser_roi_par_discipline_sql) ---
from utils.database import (
    init_db, migrer_anciens_json_vers_sqlite, charger_historique, 
    sauvegarder_historique, charger_courses_jour_db, sauvegarder_courses_jour_db, 
    load_bets_from_db, update_bet_status, get_conn, DB_PATH, save_bet_to_db,
    save_multiple_bets_to_db, get_bankroll, analyser_roi_par_discipline_sql,
    analyser_bilan_journalier_sql
)
from utils.api_pmu import telecharger_pmu_date, safe_float, HEADERS
from utils.ai_model import (
    charger_modele_ia, sauvegarder_modele_ia, calculer_parametres_adaptatifs,
    evaluer_score_cheval, normaliser_scores_chevaux, calculer_valeur_esperee_avancee,
    generer_plan_budget_journalier, retroaction_apprentissage_ia,
    rafraichir_modele_ml, optimiser_poids_ia_automatique,
    simuler_et_entrainer_journee_intelligente
)

# --- CONFIGURATION DES LOGS (Doit être en premier) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PMU_Pro")

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Analyse & Stratégie PMU Pro + IA",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DOSSIER = Path(".")

# Initialisation DB & Tables SQL
init_db()

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

# =====================================================================
# SAS UNIQUE OBLIGATOIRE : PASSAGE OBLIGATOIRE PAR L'IA AVANT ENREGISTREMENT
# =====================================================================
def enregistrer_pari_via_ia(date_iso, reunion, course_num, discipline, cheval_data, type_pari_souhaite, mise_totale):
    """
    Sas unique optimisé pour l'automatisation en masse : 
    Insère directement le pari en base sans recharger ni parcourir tout l'historique.
    """
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
        discipline=discipline
    )
    return True

# --- FONCTIONS UTILITAIRES POUR LES STATISTIQUES (ROI) ---
def test_student_yield(paris_regles, mu_0=-0.15):
    rendements = []
    for pari_item in paris_regles:
        m = safe_float(pari_item.get("mise", 0))
        g = safe_float(pari_item.get("gain", 0))
        if m > 0:
            rendements.append((g - m) / m)
    n = len(rendements)
    if n < 2:
        return None, None
    mean_r = sum(rendements) / n
    variance = sum((r - mean_r) ** 2 for r in rendements) / (n - 1)
    s = math.sqrt(variance)
    if s == 0:
        return None, None
    t_score = (mean_r - mu_0) / (s / math.sqrt(n))
    return t_score, mean_r

def test_significativite_monte_carlo(paris_regles, iterations=10000):
    if len(paris_regles) < 10:
        return None, None
    profit_reel = sum(safe_float(pari_item.get("gain", 0)) - safe_float(pari_item.get("mise", 0)) for pari_item in paris_regles)
    profits_nets = [safe_float(pari_item.get("gain", 0)) - safe_float(pari_item.get("mise", 0)) for pari_item in paris_regles]
    profits_array = np.array(profits_nets)
    n_paris = len(profits_array)
    if n_paris == 0:
        return None, None
    succes = 0
    for _ in range(iterations):
        chantillon_simule = np.random.choice(profits_array, size=n_paris, replace=True)
        if np.sum(chantillon_simule) >= profit_reel:
            succes += 1
    return succes / iterations, profit_reel

def verifier_resultats_automatiques_pmu(historique):
    modifie = False
    
    paris_en_attente = []
    for pari_item in historique:
        if str(pari_item.get("statut", "")).strip().lower() == "en attente":
            paris_en_attente.append(pari_item)

    total_a_verifier = len(paris_en_attente)
    if total_a_verifier == 0:
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

    progress_bar = st.progress(0, text="Vérification des résultats PMU...")
    nb_traites = 0

    for cle, paris_groupe in groupes_paris.items():
        date_pmu, reunion_str, course_str = cle
        
        try:
            url_part = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}/{reunion_str}/{course_str}/participants"
            res_part = requests.get(url_part, headers=HEADERS, timeout=10)
            if res_part.status_code != 200:
                continue

            liste_partants_bruts = res_part.json().get("participants", [])
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

            limite_places = 3 if len(liste_partants_bruts) >= 8 else 2

            for pari_item in paris_groupe:
                nb_traites += 1
                progress_bar.progress(min(1.0, nb_traites / total_a_verifier), text=f"Vérification pari {nb_traites}/{total_a_verifier} ({pari_item.get('date', '')})...")

                details = str(pari_item.get("details", pari_item.get("cheval", "")))
                mise_totale = safe_float(pari_item.get("mise", 0))
                gain_total, un_gagne = 0.0, False
                parts = details.split("|") if "|" in details else [details]

                for p_ligne in parts:
                    p_lower = p_ligne.lower()
                    nums_part = re.findall(r"N°\s*(\d+)", p_ligne)
                    mise_part_m = re.search(r"\((\d+(?:[\.,]\d+)?)\s*€\)", p_ligne)
                    mise_part = float(mise_part_m.group(1).replace(",", ".")) if mise_part_m else (mise_totale / len(parts))

                    if ("placé" in p_lower or "place" in p_lower or "sécu" in p_lower) and nums_part:
                        num_secu = str(nums_part[0])
                        cote_direct = cotes_reelles.get(num_secu, 3.0)
                        div_ref = max(1.1, 1.0 + (cote_direct - 1.0) / (3.6 if len(liste_partants_bruts) >= 8 else 2.5))
                        if num_secu in arrivee_trouvee[:limite_places]:
                            gain_total += mise_part * div_ref
                            un_gagne = True
                    elif ("gagnant" in p_lower or "poker" in p_lower or "simple" in p_lower) and nums_part:
                        num_poker = str(nums_part[0])
                        div_ref = cotes_reelles.get(num_poker, 3.0)
                        if arrivee_trouvee and num_poker == arrivee_trouvee[0]:
                            gain_total += mise_part * div_ref
                            un_gagne = True

                pari_item["statut"] = "Gagné" if un_gagne else "Perdu"
                pari_item["gain"] = round(gain_total, 2)
                pari_item["diagnostic"] = retroaction_apprentissage_ia(pari_item, arrivee_trouvee, cotes_reelles, liste_partants_bruts)
                
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
            optimiser_poids_ia_automatique()
            logger.info("Optimisation automatique des poids IA exécutée suite à la mise à jour des résultats.")
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation automatique : {e}")

    return modifie


def analyser_bilan_journalier_sql():
    df = load_bets_from_db()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df_regle = df[df["statut"].isin(["Gagné", "Perdu"])].copy()
    if df_regle.empty:
        return pd.DataFrame()
    df_regle["mise"] = pd.to_numeric(df_regle["mise"], errors="coerce").fillna(0)
    df_regle["gain"] = pd.to_numeric(df_regle["gain"], errors="coerce").fillna(0)
    
    grouped = df_regle.groupby("date").agg(
        nb_paris=("id", "count"),
        total_mises=("mise", "sum"),
        total_gains=("gain", lambda x: df_regle.loc[x.index, "gain"][df_regle.loc[x.index, "statut"] == "Gagné"].sum())
    ).reset_index()
    
    grouped["bilan_net"] = grouped["total_gains"] - grouped["total_mises"]
    grouped["roi_pourcent"] = grouped.apply(lambda row: (row["bilan_net"] / row["total_mises"] * 100) if row["total_mises"] > 0 else 0, axis=1)
    grouped = grouped.sort_values(by="date", ascending=False) if "date" in grouped.columns else grouped
    grouped.columns = ["Date", "Nb Paris", "Total Mises (€)", "Total Gains (€)", "Bilan Net (€)", "ROI (%)"]
    return grouped

# --- NAVIGATION PAR ONGLETS EN HAUT ---
tab_accueil, tab_chrono, tab_analyse, tab_ia, tab_suivi, tab_bilan, tab_admin = st.tabs([
    "🏠 Accueil",
    "⏰ Chrono des Courses",
    "📊 Analyse & Value Bets",
    "🧠 Moteur IA",
    "📈 Suivi & ROI",
    "🏟️ Bilan Réunion",
    "🛠️ Administration"
])

# ================= 1. ACCUEIL =================
with tab_accueil:
    st.title("🐎 Bienvenue sur Bilan Réunion Pro + IA")
    st.markdown("Tableau de bord centralisé pour l'analyse des courses hippiques et l'optimisation des paris.")
    
    # ➔ Affichage dynamique de la Bankroll actuelle
    bankroll_actuelle = get_bankroll()
    st.info(f"💰 **Bankroll Actuelle :** `{bankroll_actuelle:.2f} €`")

    historique = charger_historique()
    total_paris = len(historique)
    gagnes = sum(1 for pari_item in historique if pari_item.get("statut") == "Gagné")
    mises_totales = sum(safe_float(pari_item.get("mise", 0)) for pari_item in historique)
    gains_totaux = sum(safe_float(pari_item.get("gain", 0)) for pari_item in historique if pari_item.get("statut") == "Gagné")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Paris Enregistrés", total_paris)
    col2.metric("Paris Gagnants", gagnes)
    col3.metric("Mises Totales", f"{mises_totales:.2f} €")
    col4.metric("Bilan Net", f"{gains_totaux - mises_totales:.2f} €", delta=f"{gains_totaux - mises_totales:.2f} €")

    st.divider()
    
    st.subheader("🧠 Évolution de la Performance de l'IA (Progression & Fiabilité)")
    
    paris_regles = [p for p in historique if str(p.get("statut", "")).strip().lower() in ["gagné", "perdu"]]
    
    if len(paris_regles) >= 5:
        df_ia = pd.DataFrame(paris_regles)
        df_ia["date"] = pd.to_datetime(df_ia["date"], errors="coerce")
        df_ia = df_ia.dropna(subset=["date"]).sort_values("date")
        
        df_ia["est_gagne"] = df_ia["statut"].apply(lambda s: 1 if str(s).strip().lower() == "gagné" else 0)
        df_ia["Taux de Réussite Cumulé (%)"] = (df_ia["est_gagne"].expanding().mean() * 100).round(1)
        
        st.markdown("##### 📈 Courbe du Taux de Réussite Global de l'IA au fil du temps (%)")
        st.line_chart(df_ia.set_index("date")["Taux de Réussite Cumulé (%)"], width="stretch")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            df_ia["mise_num"] = df_ia["mise"].apply(safe_float)
            df_ia["gain_num"] = df_ia["gain"].apply(safe_float)
            df_ia["profit_cumul"] = (df_ia["gain_num"] - df_ia["mise_num"]).cumsum()
            df_ia["mises_cumul"] = df_ia["mise_num"].cumsum()
            df_ia["ROI Cumulé (%)"] = ((df_ia["profit_cumul"] / df_ia["mises_cumul"]) * 100).fillna(0).round(1)
            
            st.markdown("##### 📊 Évolution du ROI de l'IA (%)")
            st.line_chart(df_ia.set_index("date")["ROI Cumulé (%)"], width="stretch")
            
        with col_p2:
            st.info("""
            **Comment lire cette progression :**
            * **Le Taux de Réussite Cumulé (%)** montre si l'IA cible de mieux en mieux les résultats au fil du temps.
            * **Le ROI Cumulé (%)** indique si la rentabilité globale s'améliore.
            """)
    else:
        st.info("Il faut au moins 5 paris réglés (Gagné/Perdu) pour afficher la courbe de progression de l'IA.")

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
                donnees_chrono, _ = charger_courses_jour_db(date_chrono_iso)
                st.session_state["cache_chrono_donnees"] = donnees_chrono
                st.session_state["cache_chrono_date"] = date_chrono_iso
            else:
                st.warning("Aucune donnée récupérée pour cette date.")

    if btn_charger or st.session_state.get("cache_chrono_date") == date_chrono_iso:
        if "cache_chrono_donnees" not in st.session_state or st.session_state.get("cache_chrono_date") != date_chrono_iso:
            with st.spinner("Chargement des courses depuis la base..."):
                donnees_chrono, _ = charger_courses_jour_db(date_chrono_iso)
                st.session_state["cache_chrono_donnees"] = donnees_chrono
                st.session_state["cache_chrono_date"] = date_chrono_iso
        else:
            donnees_chrono = st.session_state["cache_chrono_donnees"]

        if donnees_chrono:
            toutes_courses = []
            for c_elem in donnees_chrono:
                nom_c = str(c_elem.get("nom_course", "")).strip()
                if not nom_c or nom_c.isdigit() or len(nom_c) <= 2:
                    continue

                r_nom = f"{c_elem.get('reunion', 'R1')} - {c_elem.get('hippodrome', 'HIPPODROME')}"
                chevaux_c = c_elem.get("chevaux", [])
                chevaux_val_c = [c for c in chevaux_c if safe_float(c.get("cote")) > 1.0 or c.get("cote") is None]

                base_chev, poker_chev = {"num": "?", "nom": "Inconnu", "cote": 0.0, "ev_index": 0.0}, {"num": "?", "nom": "Inconnu", "cote": 0.0, "ev_index": 0.0}
                
                if chevaux_val_c:
                    params_ad_chrono = calculer_parametres_adaptatifs()
                    for c in chevaux_val_c:
                        c["score_analyse"] = evaluer_score_cheval(
                            c, c_elem.get("discipline"), c_elem.get("terrain_officiel"),
                            c_elem.get("corde", "Corde standard"), date_chrono_iso, params_ad_chrono, 
                            distance_course=c_elem.get("distance", ""),
                            hippodrome=c_elem.get("hippodrome", ""),
                            utiliser_ml=True
                        )
                    normaliser_scores_chevaux(chevaux_val_c, "score_analyse")
                    calculer_valeur_esperee_avancee(chevaux_val_c, len(chevaux_c))

                    chevaux_val_c.sort(key=lambda x: (x.get("ev_index", 0), x["score_analyse"]), reverse=True)
                    
                    candidats_valides = [c for c in chevaux_val_c if safe_float(c.get("ev_index", 0)) >= 3.80 or safe_float(c.get("ev_place_index", 0)) >= 2.50]
                    
                    if candidats_valides:
                        base_chev = candidats_valides[0]
                        outsiders_c = [c for c in candidats_valides if 7.0 <= safe_float(c.get("cote")) <= 12.0 and c["num"] != base_chev["num"]]
                        poker_chev = max(outsiders_c, key=lambda x: x.get("ev_index", 0)) if outsiders_c else (candidats_valides[1] if len(candidats_valides) > 1 else base_chev)
                        
                        toutes_courses.append({
                            "heure": c_elem.get("heure", "13:30"),
                            "reunion": r_nom,
                            "course_num": c_elem.get("course", "C1"),
                            "nom_course": nom_c,
                            "discipline": c_elem.get("discipline", ""),
                            "data": c_elem,
                            "base": base_chev,
                            "poker": poker_chev,
                        })

            toutes_courses.sort(key=lambda x: x["heure"])

            st.success(f"{len(toutes_courses)} course(s) chargée(s) avec des opportunités valides pour le {date_chrono_iso}.")
            
            for idx_c, item_c in enumerate(toutes_courses):
                course_obj = item_c["data"]
                b_chev = item_c["base"]
                p_chev = item_c["poker"]
                cle_unique_course = f"{item_c['reunion']}_{item_c['course_num']}_{idx_c}"

                with st.expander(f"🕒 {item_c['heure']} | {item_c['reunion']} ➔ {item_c['course_num']} : {item_c['nom_course']}"):
                    st.markdown(f"**Base Value Bet (Sécurité) :** Simple Placé ➔ N°{b_chev.get('num')} - {b_chev.get('nom')} (Cote Placé: {b_chev.get('cote_place_estimee', 0):.1f} | **EV Placé: {b_chev.get('ev_place_index', 0):.2f}**)")
                    st.markdown(f"**Coup de Poker Value :** Simple Gagnant ➔ N°{p_chev.get('num')} - {p_chev.get('nom')} (Cote: {safe_float(p_chev.get('cote')):.1f} | **EV: {p_chev.get('ev_index', 0):.2f}**)")

                    col_m, col_b = st.columns([2, 1])
                    with col_m:
                        mise_input = st.number_input("Mise Totale (€)", min_value=1, value=10, key=f"m_{cle_unique_course}")
                        mise_secu = round(mise_input * 0.8, 1)
                        mise_poker = round(mise_input - mise_secu, 1)
                        st.caption(f"💡 Répartition Sécurisée : **{mise_secu} €** Sécu (80%) | **{mise_poker} €** Poker (20%)")

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
                                mise_totale=mise_secu
                            )
                            if mise_poker > 0:
                                enregistrer_pari_via_ia(
                                    date_iso=date_chrono_iso,
                                    reunion=item_c["reunion"],
                                    course_num=item_c["course_num"],
                                    discipline=course_obj.get("discipline"),
                                    cheval_data=p_chev,
                                    type_pari_souhaite="Simple Gagnant (Poker Value)",
                                    mise_totale=mise_poker
                                )
                            st.success("Paris validés, analysés et enregistrés via l'IA !")
                            st.rerun()
        else:
            st.info("Aucune donnée enregistrée en base pour cette date. Cliquez sur 'Télécharger/Actualiser' si nécessaire.")
    else:
        st.info("Sélectionnez une date puis cliquez sur **'Afficher les courses'** pour charger le programme de la journée.")

# ================= 3. ANALYSE & VALUE BETS =================
with tab_analyse:
    st.subheader("📊 Analyse Intégrale & Détection de Value Bets")
    date_sel = st.date_input("Date du jour", key="date_analyse_picker")
    date_iso = date_sel.strftime("%Y-%m-%d")

    donnees, reunions_map = charger_courses_jour_db(date_iso)
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
            type_pari_sel = st.selectbox("Type de pari", ["Simple Gagnant", "Simple Placé", "Value Bet IA"], key="sel_type_pari_analyse")
        with col_cfg2:
            mise_analyse = st.number_input("Mise (€)", min_value=1.0, value=10.0, step=1.0, key="mise_analyse_input")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        if st.button("⚡ Lancer l'Analyse Maximisation Gains IA", key="btn_lancer_analyse_ia"):
            params = calculer_parametres_adaptatifs()
            for c in course_curr.get("chevaux", []):
                c["score_ia"] = evaluer_score_cheval(
                    c, course_curr.get("discipline"), course_curr.get("terrain_officiel"),
                    course_curr.get("corde", "Corde standard"), date_iso, params, distance_course=course_curr.get("distance", "")
                )
            normaliser_scores_chevaux(course_curr.get("chevaux", []), "score_ia")
            calculer_valeur_esperee_avancee(course_curr.get("chevaux", []), len(course_curr.get("chevaux", [])))

            chevaux_tries = sorted(course_curr.get("chevaux", []), key=lambda x: x.get("ev_index", 0), reverse=True)
            st.session_state["chevaux_analyse_courant"] = chevaux_tries

            st.dataframe([{
                "N°": c["num"], "Nom": c["nom"], "Driver": c.get("driver"), "Cote": c.get("cote"),
                "Score IA": c.get("score_ia"), "Proba (Softmax)": f"{safe_float(c.get('proba_estimee',0))*100:.1f}%",
                "EV Gagnant": c.get("ev_index"), "EV Placé": c.get("ev_place_index")
            } for c in chevaux_tries], width="stretch")

        if "chevaux_analyse_courant" in st.session_state and st.session_state["chevaux_analyse_courant"]:
            st.divider()
            st.markdown("### 📝 Enregistrer un pari sur cette course analysée")
            chevaux_analyse = st.session_state["chevaux_analyse_courant"]
            
            options_chevaux = [f"N°{c['num']} - {c['nom']} (Cote: {safe_float(c.get('cote')):.1f})" for c in chevaux_analyse]
            cheval_selectionne_str = st.selectbox("Sélectionner le cheval", options_chevaux, key="sel_cheval_analyse")
            
            st.info(f"Configuration active : **{type_pari_sel}** | Mise : **{mise_analyse} €**")
                
            if st.button("⚡ Valider & Enregistrer ce Pari", key="btn_valider_analyse_cours"):
                match_num = re.search(r"N°(\d+)", cheval_selectionne_str)
                num_cheval = match_num.group(1) if match_num else "?"
                
                cheval_obj = next((c for c in chevaux_analyse if str(c.get("num")) == str(num_cheval)), chevaux_analyse[0])
                
                enregistrer_pari_via_ia(
                    date_iso=date_iso,
                    reunion=reunion_choisie,
                    course_num=course_curr.get("course"),
                    discipline=course_curr.get("discipline"),
                    cheval_data=cheval_obj,
                    type_pari_souhaite=type_pari_sel,
                    mise_totale=mise_analyse
                )
                st.success("Pari validé et enregistré avec succès via l'IA depuis l'onglet Analyse !")
                st.rerun()

        st.divider()
        st.subheader("💰 Allocation Stratégique (Critère de Kelly)")
        budget_saisi = st.number_input("Budget Global à Allouer (€)", min_value=5, max_value=500, value=50, step=5, key="budget_kelly")

        if st.button("🎲 Calculer le Plan d'Allocation Optimal", key="btn_calc_kelly"):
            params_ad = calculer_parametres_adaptatifs()
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
                    
                    details_str = f"{type_pari} ➔ N°{num_cheval} {cheval_nom} (Cote: {cote} | EV: {max(safe_float(row.get('EV', 0)), 0):.2f})"
                    
                    paris_a_inserer.append({
                        "date": date_iso,
                        "reunion": reunion,
                        "course": course_num,
                        "cheval": details_str,
                        "cote": cote,
                        "mise": mise,
                        "statut": "En attente",
                        "gain": 0.0,
                        "discipline": course_curr.get("discipline", "Galop Plat")
                    })
                
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
    modele = charger_modele_ia()

    if "disciplines_specifiques" not in modele:
        modele["disciplines_specifiques"] = {}

    disciplines_disponibles = ["Trot Attelé", "Galop Plat", "Haies", "Steeple-chase", "Trot Monté"]
    discipline_selectionnee = st.selectbox("🎯 Sélectionner la discipline à configurer", disciplines_disponibles, key="select_discipline_ia")

    disc_dict = modele["disciplines_specifiques"].get(discipline_selectionnee, {})

    col_ia1, col_ia2 = st.columns(2)
    with col_ia1:
        st.subheader(f"⚙️ Poids Heuristiques : {discipline_selectionnee}")
        with st.form(f"form_poids_ia_{discipline_selectionnee}"):
            poids_musique = st.slider("Poids Musique", 0.1, 10.0, float(disc_dict.get("poids_musique", modele.get("poids_musique", 1.05))), 0.05)
            poids_ferrage = st.slider("Poids Ferrage", 0.1, 10.0, float(disc_dict.get("poids_ferrage", modele.get("poids_ferrage", 1.25))), 0.05)
            poids_terrain = st.slider("Poids Terrain", 0.1, 10.0, float(disc_dict.get("poids_terrain", modele.get("poids_terrain", 1.15))), 0.05)
            poids_poids = st.slider("Poids Poids", 0.1, 10.0, float(disc_dict.get("poids_poids", modele.get("poids_poids", 1.0))), 0.05)
            poids_cote_tendance = st.slider("Poids Tendance Cote", 0.1, 10.0, float(disc_dict.get("poids_cote_tendance", modele.get("poids_cote_tendance", 1.35))), 0.05)
            poids_driver = st.slider("Poids Driver", 0.1, 10.0, float(disc_dict.get("poids_driver", modele.get("poids_driver", 1.15))), 0.05)
            poids_corde = st.slider("Poids Corde", 0.1, 10.0, float(disc_dict.get("poids_corde", modele.get("poids_corde", 1.0))), 0.05)
            poids_hippodrome = st.slider("Poids Hippodrome/Acteur", 0.1, 10.0, float(disc_dict.get("poids_hippodrome_acteur", modele.get("poids_hippodrome_acteur", 1.25))), 0.05)
            poids_distance = st.slider("Poids Distance", 0.1, 10.0, float(disc_dict.get("poids_distance", modele.get("poids_distance", 1.10))), 0.05)
            poids_outsider_cache = st.slider("Poids Outsiders Cachés", 0.1, 10.0, float(disc_dict.get("poids_outsider_cache", modele.get("poids_outsider_cache", 1.30))), 0.05)
            
            st.markdown("---")
            st.markdown("##### 🎯 Filtres de Cotes par Discipline")
            min_cote_input = st.number_input("Cote Minimale Filtrée", 1.0, 50.0, float(disc_dict.get("min_cote", 2.20)), 0.10)
            max_cote_input = st.number_input("Cote Maximale Filtrée", 1.0, 50.0, float(disc_dict.get("max_cote", 8.50)), 0.10)
            
            seuil_ev = st.number_input("Seuil Minimal Value Bet (EV)", 1.0, 10.0, float(modele.get("seuil_value_bet", 3.50)), 0.05)
            bloque_disc = st.checkbox("🛑 Bloquer les paris sur cette discipline (Quarantaine)", value=disc_dict.get("bloque_paris", False))
            
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
                    "bloque_paris": bloque_disc
                }
                modele["seuil_value_bet"] = seuil_ev
                
                sauvegarder_modele_ia(modele)
                st.success(f"Modèle mis à jour avec succès pour le {discipline_selectionnee} !")
                st.rerun()

    with col_ia2:
        st.subheader(f"📊 Statistiques d'Impact : {discipline_selectionnee}")
        
        # Filtrage dynamique des paris de la discipline sélectionnée depuis la base
        df_all_bets = load_bets_from_db()
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
        
        # Statistiques globales du modèle (ou conservées globalement)
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
        
        params_actuels = calculer_parametres_adaptatifs()
        st.info(f"💡 Message Modèle Adaptatif : {params_actuels.get('message_auto')}")

# ================= 5. SUIVI & ROI FINANCIER =================
with tab_suivi:
    st.subheader("📈 Suivi Financier, ROI & Progression Course par Course")
    if st.button("🔄 Vérifier Automatiquement les Résultats PMU", key="btn_verif_pmu_auto"):
        hist = charger_historique()
        if verifier_resultats_automatiques_pmu(hist):
            sauvegarder_historique(hist)
            st.success("Résultats PMU mis à jour !")
            st.rerun()
        else:
            st.info("Aucun résultat supplémentaire trouvé.")

    historique = charger_historique()
    paris_regles = [pari_item for pari_item in historique if str(pari_item.get("statut", "")).strip().lower() in ["gagné", "perdu", "remboursé"]]
    
    total_mises = sum(safe_float(pari_item.get("mise", 0)) for pari_item in paris_regles)
    total_gains = sum(safe_float(pari_item.get("gain", 0)) for pari_item in paris_regles if str(pari_item.get("statut", "")).strip().lower() in ["gagné", "remboursé"])
    bilan_net = total_gains - total_mises
    roi = (bilan_net / total_mises * 100) if total_mises > 0 else 0.0
    paris_valides_win = [p for p in paris_regles if str(p.get("statut", "")).strip().lower() in ["gagné", "perdu"]]
    taux_win = (sum(1 for pari_item in paris_valides_win if str(pari_item.get("statut", "")).strip().lower() == "gagné") / len(paris_valides_win) * 100) if paris_valides_win else 0.0

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
        df_chrono["ROI Cumulé (%)"] = ((df_chrono["Bilan Net Cumulé (€)"] / df_chrono["Mises Cumulées (€)"]) * 100).fillna(0).round(2)
        df_chrono["Index Course"] = [f"Pari #{i+1}" for i in range(len(df_chrono))]
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### 📈 Courbe du Bilan Net Cumulé (€) au fil des courses")
            st.line_chart(df_chrono.set_index("Index Course")["Bilan Net Cumulé (€)"], width="stretch")
            
        with col_g2:
            st.markdown("##### 📊 Courbe du ROI Cumulé (%) au fil des courses")
            st.line_chart(df_chrono.set_index("Index Course")["ROI Cumulé (%)"], width="stretch")
            
        with st.expander("🔍 Voir le détail du tableau chronologique des paris", expanded=False):
            st.dataframe(df_chrono[["date", "reunion", "course", "cheval", "mise", "statut", "gain", "Bilan Net Cumulé (€)", "ROI Cumulé (%)"]], width="stretch")
    else:
        st.info("Il faut au moins 2 paris réglés (Gagné/Perdu) pour afficher l'évolution chronologique course par course.")

    st.divider()
    st.subheader("🧪 Tests de Significativité Statistique")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t_score, mean_r = test_student_yield(paris_regles)
        if t_score is not None:
            st.write(f"**Test t de Student :**")
            st.write(f"- Rendement Moyen : `{mean_r*100:.2f}%` | t-score : `{t_score:.2f}`")
        else:
            st.info("Pas assez de paris pour le test t (min. 2).")

    with col_t2:
        p_val, prof_r = test_significativite_monte_carlo(paris_regles)
        if p_val is not None:
            st.write(f"**Simulation Monte Carlo :**")
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
        # 'use_container_width=True' permet d'afficher proprement toutes les colonnes dynamiques
        st.dataframe(df_sql, use_container_width=True)
    else:
        st.info("Aucune donnée disponible pour l'analyse par discipline.")

    st.divider()
    st.subheader("📅 Bilans Journaliers")
    df_journalier = analyser_bilan_journalier_sql()
    if not df_journalier.empty:
        st.dataframe(df_journalier, use_container_width=True)
    else:
        st.info("Aucun pari réglé (Gagné/Perdu) disponible pour afficher le bilan journalier.")

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
        st.subheader("🤖 Apprentissage Séquentiel (Jour par Jour) & Itératif vers ROI Positif")
        st.markdown("L'IA traite les courses **jour après jour**, vérifie les résultats et **s'adapte immédiatement chaque jour**, avec un suivi détaillé de chaque étape.")
        
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
                
                st.markdown("### 📊 Suivi en direct de l'Apprentissage Séquentiel")
                col_suivi1, col_suivi2 = st.columns(2)
                with col_suivi1:
                    box_statut = st.empty()
                with col_suivi2:
                    box_roi = st.empty()
                
                box_details = st.empty()
                
                st.markdown("#### 📈 Évolution du ROI au fil des jours et des tentatives :")
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
                    
                    for i in range(nb_jours):
                        courante_iso = courante_dt.strftime("%Y-%m-%d")
                        box_details.info(f"📅 [Tentative {tentative}] Traitement et paris pour la journée : **{courante_iso}**")
                        
                        try:
                            try:
                                conn_net = get_conn()
                                cur_net = conn_net.cursor()
                                cur_net.execute("DELETE FROM pmu_bets WHERE date = %s", (courante_iso,))
                                conn_net.commit()
                                cur_net.close()
                                conn_net.close()
                            except Exception as e_net:
                                logger.error(f"Erreur lors du nettoyage préventif des paris pour la date {courante_iso} : {e_net}")

                            nb_c_jour, mises_jour = simuler_et_entrainer_journee_intelligente(courante_iso, float(mise_auto_defaut))
                            total_paris_ajoutes += nb_c_jour
                            gains_jour = 0.0
                                
                            box_details.info(f"🔍 [Tentative {tentative}] Vérification PMU et **adaptation instantanée de l'IA** pour le {courante_iso}...")
                            hist_actuel = charger_historique()
                            if verifier_resultats_automatiques_pmu(hist_actuel):
                                sauvegarder_historique(hist_actuel)
                            
                            optimiser_poids_ia_automatique()
                            
                            hist_verifie = charger_historique()
                            for p_item in hist_verifie:
                                if str(p_item.get("date", ""))[:10] == courante_iso:
                                    if p_item.get("statut") in ["Gagné", "Remboursé"]:
                                        gains_jour += safe_float(p_item.get("gain", 0.0))
                            
                            mises_totales_periode += mises_jour
                            gains_totaux_periode += gains_jour
                            
                            roi_cumule_periode = ((gains_totaux_periode - mises_totales_periode) / mises_totales_periode * 100) if mises_totales_periode > 0 else 0.0

                            historique_journalier.append({
                                "Tentative": f"Tentative {tentative}",
                                "Date": courante_iso,
                                "Paris ajoutés (jour)": nb_c_jour,
                                "ROI Global Cumulé (%)": round(roi_cumule_periode, 2)
                            })
                            table_suivi_container.dataframe(pd.DataFrame(historique_journalier), use_container_width=True)
                                
                        except Exception as e:
                            logger.error(f"Erreur date {courante_iso} : {e}")
                            
                        courante_dt += delta
                    
                    roi_global = ((gains_totaux_periode - mises_totales_periode) / mises_totales_periode * 100) if mises_totales_periode > 0 else 0.0
                        
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
        st.markdown("Supprimez automatiquement les paris restés **'En attente'** dont la date est trop ancienne et qui n'ont pas de résultat.")
        
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
                    cursor.execute("DELETE FROM pmu_bets WHERE statut = 'En attente' AND date < %s", (date_limite_str,))
                    nb_supprimes = cursor.rowcount
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    hist = charger_historique()
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
        st.markdown("Cette action supprimera définitivement **tous les paris enregistrés, les courses en cache, l'historique et les modèles** de la base de données.")
        
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
                    
                    st.success("🗑️ Remise à zéro effectuée avec succès ! Toutes les données ont été supprimées.")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la remise à zéro : {e}")
