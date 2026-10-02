# app.py
import datetime
import json
import logging
from pathlib import Path
import math
import re
import requests
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from utils.ai_model import optimiser_poids_ia_automatique

# --- CONFIGURATION DES LOGS (Doit être en premier) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PMU_Pro")

from utils.database import init_db, migrer_anciens_json_vers_sqlite, charger_historique, sauvegarder_historique, charger_courses_jour_db, DB_PATH
from utils.api_pmu import telecharger_pmu_date, safe_float, HEADERS
from utils.ai_model import (
    charger_modele_ia, sauvegarder_modele_ia, calculer_parametres_adaptatifs,
    evaluer_score_cheval, normaliser_scores_chevaux, calculer_valeur_esperee_avancee,
    generer_plan_budget_journalier, retroaction_apprentissage_ia
)

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Analyse & Stratégie PMU Pro + IA",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DOSSIER = Path(".")

# Initialisation DB
init_db()
migrer_anciens_json_vers_sqlite()

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

# --- FONCTIONS UTILITAIRES POUR LES STATISTIQUES (ROI) ---
def test_student_yield(paris_regles, mu_0=-0.15):
    rendements = []
    for p in paris_regles:
        m = safe_float(p.get("mise", 0))
        g = safe_float(p.get("gain", 0))
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
    profit_reel = sum(safe_float(p.get("gain", 0)) - safe_float(p.get("mise", 0)) for p in paris_regles)
    profits_nets = [safe_float(p.get("gain", 0)) - safe_float(p.get("mise", 0)) for p in paris_regles]
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
    dates_modifiees = set()
    paris_en_attente = [p for p in historique if p.get("statut") == "En attente"]
    total_a_verifier = len(paris_en_attente)
    if total_a_verifier == 0:
        return False

    progress_bar = st.progress(0, text="Vérification des résultats PMU...")
    i = 0
    for p in historique:
        if p.get("statut") == "En attente":
            i += 1
            progress_bar.progress(min(1.0, i / total_a_verifier), text=f"Vérification pari {i}/{total_a_verifier}...")
            date_pari = str(p.get("date", "")).strip()
            reunion_raw = str(p.get("reunion", "")).strip()
            course_raw = str(p.get("course_num", "")).strip()
            course_full = str(p.get("course", "")).strip()

            date_pmu, date_iso_norm = None, date_pari
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    dt = datetime.datetime.strptime(date_pari, fmt)
                    date_pmu = dt.strftime("%d%m%Y")
                    date_iso_norm = dt.strftime("%Y-%m-%d")
                    break
                except Exception:
                    pass
            if not date_pmu:
                continue

            texte_global = f"{reunion_raw} {course_raw} {course_full}"
            r_match = re.search(r"R\s*(\d+)", texte_global, re.IGNORECASE)
            reunion_str = f"R{r_match.group(1)}" if r_match else ""
            c_match = re.search(r"C\s*(\d+)", texte_global, re.IGNORECASE)
            if not c_match:
                c_match = re.search(r"\b(\d+)(?:ère|ème|e)?\s*course\b", texte_global, re.IGNORECASE)
            course_str = f"C{c_match.group(1)}" if c_match else ""

            if not reunion_str or not course_str:
                continue

            try:
                res_rap = requests.get(f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}/{reunion_str}/{course_str}/rapports", headers=HEADERS, timeout=10)
                res_part = requests.get(f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_pmu}/{reunion_str}/{course_str}/participants", headers=HEADERS, timeout=10)
                if res_part.status_code != 200:
                    continue

                liste_partants_bruts = res_part.json().get("participants", [])
                cotes_reelles, partants_arrives = {}, []
                for part in liste_partants_bruts:
                    num_pmu = str(part.get("numPmu"))
                    rapport = part.get("dernierRapportDirect")
                    if isinstance(rapport, dict) and isinstance(rapport.get("rapport"), (int, float)):
                        cotes_reelles[num_pmu] = float(rapport.get("rapport"))
                    ordre = part.get("ordreArrivee")
                    if isinstance(ordre, int) and ordre > 0:
                        partants_arrives.append((ordre, num_pmu))

                partants_arrives.sort(key=lambda x: x[0])
                arrivee_trouvee = [num for _, num in partants_arrives]
                if not arrivee_trouvee:
                    continue

                details = str(p.get("details", ""))
                mise_totale = safe_float(p.get("mise", 0))
                gain_total, un_gagne = 0.0, False
                parts = details.split("|") if "|" in details else [details]
                limite_places = 3 if len(liste_partants_bruts) >= 8 else 2

                for part in parts:
                    part_lower = part.lower()
                    nums_part = re.findall(r"N°\s*(\d+)", part)
                    mise_part_m = re.search(r"\((\d+(?:[\.,]\d+)?)\s*€\)", part)
                    mise_part = float(mise_part_m.group(1).replace(",", ".")) if mise_part_m else (mise_totale / len(parts))

                    if ("placé" in part_lower or "place" in part_lower or "sécu" in part_lower) and nums_part:
                        num_secu = str(nums_part[0])
                        div_ref = max(1.1, 1.0 + (cotes_reelles.get(num_secu, 3.0) - 1.0) / (3.6 if len(liste_partants_bruts) >= 8 else 2.5))
                        if num_secu in arrivee_trouvee[:limite_places]:
                            gain_total += mise_part * div_ref
                            un_gagne = True
                    elif ("gagnant" in part_lower or "poker" in part_lower) and nums_part:
                        num_poker = str(nums_part[0])
                        div_ref = cotes_reelles.get(num_poker, 3.0)
                        if num_poker == arrivee_trouvee[0]:
                            gain_total += mise_part * div_ref
                            un_gagne = True

                p["statut"] = "Gagné" if un_gagne else "Perdu"
                p["gain"] = round(gain_total, 2)
                p["diagnostic"] = retroaction_apprentissage_ia(p, arrivee_trouvee, cotes_reelles, liste_partants_bruts)
                
                # --- Enregistrement de l'arrivée pour l'apprentissage LightGBM ---
                try:
                    conn_db = sqlite3.connect(DB_PATH)
                    cur = conn_db.cursor()
                    cur.execute("SELECT data_json FROM courses_cache WHERE date_iso = ? AND reunion = ? AND course = ?", 
                                (date_iso_norm, reunion_str, course_str))
                    row_cache = cur.fetchone()
                    if row_cache:
                        race_data = json.loads(row_cache[0])
                        for part in race_data.get("chevaux", []):
                            num_pmu_str = str(part.get("num"))
                            for ord_num, p_num in partants_arrives:
                                if num_pmu_str == p_num:
                                    part["ordreArrivee"] = ord_num
                        cur.execute("UPDATE courses_cache SET data_json = ? WHERE date_iso = ? AND reunion = ? AND course = ?",
                                    (json.dumps(race_data, ensure_ascii=False), date_iso_norm, reunion_str, course_str))
                        conn_db.commit()
                    conn_db.close()
                except Exception as e:
                    logger.error(f"Erreur mise à jour ordre arrivée cache : {e}")
                # --------------------------------------------------------------------------

                modifie = True
                dates_modifiees.add(date_iso_norm)
            except Exception:
                pass  # <--- C'est cette ligne qui fermait le try principal et manquait !

    progress_bar.empty()
    return modifie

def analyser_roi_par_discipline_sql():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                discipline, COUNT(*) as nb_paris, SUM(mise) as total_mises,
                SUM(CASE WHEN statut = 'Gagné' THEN gain ELSE 0 END) as total_gains,
                (SUM(CASE WHEN statut = 'Gagné' THEN gain ELSE 0 END) - SUM(mise)) as bilan_net,
                CASE WHEN SUM(mise) > 0 THEN ((SUM(CASE WHEN statut = 'Gagné' THEN gain ELSE 0 END) - SUM(mise)) / SUM(mise)) * 100 ELSE 0 END as roi_pourcent
            FROM paris WHERE statut IN ('Gagné', 'Perdu') GROUP BY discipline ORDER BY roi_pourcent DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["Discipline", "Nb Paris", "Total Mises (€)", "Total Gains (€)", "Bilan Net (€)", "ROI (%)"])
    except Exception:
        return pd.DataFrame()

def analyser_bilan_journalier_sql():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                date as Date,
                COUNT(*) as "Nb Paris",
                SUM(mise) as "Total Mises (€)",
                SUM(CASE WHEN statut = 'Gagné' THEN gain ELSE 0 END) as "Total Gains (€)",
                (SUM(CASE WHEN statut = 'Gagné' THEN gain ELSE 0 END) - SUM(mise)) as "Bilan Net (€)",
                CASE WHEN SUM(mise) > 0 THEN ((SUM(CASE WHEN statut = 'Gagné' THEN gain ELSE 0 END) - SUM(mise)) / SUM(mise)) * 100 ELSE 0 END as "ROI (%)"
            FROM paris 
            WHERE statut IN ('Gagné', 'Perdu') 
            GROUP BY date 
            ORDER BY date DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["Date", "Nb Paris", "Total Mises (€)", "Total Gains (€)", "Bilan Net (€)", "ROI (%)"])
    except Exception:
        return pd.DataFrame()

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
    st.title("🐎 Bienvenue sur PMU Pro + IA")
    st.markdown("Tableau de bord centralisé pour l'analyse des courses hippiques et l'optimisation des paris.")
    
    historique = charger_historique()
    total_paris = len(historique)
    gagnes = sum(1 for p in historique if p.get("statut") == "Gagné")
    mises_totales = sum(safe_float(p.get("mise", 0)) for p in historique)
    gains_totaux = sum(safe_float(p.get("gain", 0)) for p in historique if p.get("statut") == "Gagné")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Paris Enregistrés", total_paris)
    col2.metric("Paris Gagnants", gagnes)
    col3.metric("Mises Totales", f"{mises_totales:.2f} €")
    col4.metric("Bilan Net", f"{gains_totaux - mises_totales:.2f} €", delta=f"{gains_totaux - mises_totales:.2f} €")

# ================= 2. CHRONO DES COURSES =================
with tab_chrono:
    st.subheader("⏰ Programme Chronologique & Sélection Value Bets")
    col_c1, col_c2 = st.columns([2, 2])
    with col_c1:
        date_chrono_sel = st.date_input("Date", key="date_chrono_picker")
        date_chrono_iso = date_chrono_sel.strftime("%Y-%m-%d")

    with col_c2:
        if st.button("📥 Télécharger/Actualiser les courses", key="btn_dl_chrono"):
            if telecharger_pmu_date(date_chrono_iso, None):
                st.success("Données actualisées et stockées en base !")
                st.rerun()

    donnees_chrono, _ = charger_courses_jour_db(date_chrono_iso)
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
                        c_elem.get("corde", "Corde standard"), date_chrono_iso, params_ad_chrono, distance_course=c_elem.get("distance", "")
                    )
                normaliser_scores_chevaux(chevaux_val_c, "score_analyse")
                calculer_valeur_esperee_avancee(chevaux_val_c, len(chevaux_c))

                chevaux_val_c.sort(key=lambda x: (x.get("ev_index", 0), x["score_analyse"]), reverse=True)
                base_chev = chevaux_val_c[0]
                outsiders_c = [c for c in chevaux_val_c if 7.0 <= safe_float(c.get("cote")) <= 22.0 and c["num"] != base_chev["num"]]
                poker_chev = max(outsiders_c, key=lambda x: x.get("ev_index", 0)) if outsiders_c else (chevaux_val_c[1] if len(chevaux_val_c) > 1 else base_chev)

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
                        nouveau_pari = {
                            "date": date_chrono_iso,
                            "reunion": item_c["reunion"],
                            "course_num": item_c["course_num"],
                            "course": f"{item_c['reunion']} - {item_c['course_num']}",
                            "discipline": course_obj.get("discipline"),
                            "type": "Rapide Value",
                            "details": f"Simple Placé (Sécurité Value) ➔ N°{b_chev.get('num', '?')} ({mise_secu}€) | Simple Gagnant (Poker Value) ➔ N°{p_chev.get('num', '?')} ({mise_poker}€)",
                            "mise": float(mise_input),
                            "statut": "En attente",
                            "gain": 0.0,
                            "diagnostic": "",
                        }
                        hist = charger_historique()
                        hist.append(nouveau_pari)
                        sauvegarder_historique(hist)
                        st.success("Pari validé et enregistré !")
                        st.rerun()
    else:
        st.info("Aucune donnée disponible pour cette date. Chargez le programme via le bouton ci-dessus.")

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
            st.dataframe([{
                "N°": c["num"], "Nom": c["nom"], "Driver": c.get("driver"), "Cote": c.get("cote"),
                "Score IA": c.get("score_ia"), "Proba (Softmax)": f"{safe_float(c.get('proba_estimee',0))*100:.1f}%",
                "EV Gagnant": c.get("ev_index"), "EV Placé": c.get("ev_place_index")
            } for c in chevaux_tries], width="stretch")

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
            st.dataframe(st.session_state["plan_courant"], width="stretch")
    else:
        st.info("Aucune réunion disponible pour cette date.")

# ================= 4. MOTEUR IA & PARAMÈTRES =================
with tab_ia:
    st.subheader("🧠 Config & Poids du Moteur IA")
    modele = charger_modele_ia()

    col_ia1, col_ia2 = st.columns(2)
    with col_ia1:
        st.subheader("⚙️ Poids Heuristiques Actuels")
        with st.form("form_poids_ia"):
            poids_musique = st.slider("Poids Musique", 0.1, 3.0, float(modele.get("poids_musique", 1.05)), 0.05)
            poids_ferrage = st.slider("Poids Ferrage", 0.1, 3.0, float(modele.get("poids_ferrage", 1.25)), 0.05)
            poids_terrain = st.slider("Poids Terrain", 0.1, 3.0, float(modele.get("poids_terrain", 1.15)), 0.05)
            poids_poids = st.slider("Poids Poids", 0.1, 3.0, float(modele.get("poids_poids", 1.0)), 0.05)
            poids_cote_tendance = st.slider("Poids Tendance Cote", 0.1, 3.0, float(modele.get("poids_cote_tendance", 1.35)), 0.05)
            poids_driver = st.slider("Poids Driver", 0.1, 3.0, float(modele.get("poids_driver", 1.15)), 0.05)
            poids_corde = st.slider("Poids Corde", 0.1, 3.0, float(modele.get("poids_corde", 1.0)), 0.05)
            poids_outsider_cache = st.slider("Poids Outsiders Cachés", 0.1, 3.0, float(modele.get("poids_outsider_cache", 1.30)), 0.05)
            
            seuil_ev = st.number_input("Seuil Minimal Value Bet (EV)", 1.0, 3.0, float(modele.get("seuil_value_bet", 1.50)), 0.05)
            
            if st.form_submit_button("💾 Sauvegarder les Poids"):
                modele["poids_musique"] = poids_musique
                modele["poids_ferrage"] = poids_ferrage
                modele["poids_terrain"] = poids_terrain
                modele["poids_poids"] = poids_poids
                modele["poids_cote_tendance"] = poids_cote_tendance
                modele["poids_driver"] = poids_driver
                modele["poids_corde"] = poids_corde
                modele["poids_outsider_cache"] = poids_outsider_cache
                modele["seuil_value_bet"] = seuil_ev
                sauvegarder_modele_ia(modele)
                st.success("Modèle mis à jour avec succès !")
                st.rerun()

    with col_ia2:
        st.subheader("📊 Statistiques d'Impact & Rétroaction")
        stats = modele.get("stats_impact", {})
        st.metric("Total Analyses IA", stats.get("total_analyses", 0))
        st.metric("Gain Cumulé IA (€)", f"{stats.get('gain_cumule_ia', 0.0):.2f} €")
        st.metric("Victoires liées au Ferrage", stats.get("victoires_par_ferrage", 0))
        
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
    st.subheader("📈 Suivi Financier, ROI & Tests Statistiques")
    if st.button("🔄 Vérifier Automatiquement les Résultats PMU", key="btn_verif_pmu_auto"):
        hist = charger_historique()
        if verifier_resultats_automatiques_pmu(hist):
            sauvegarder_historique(hist)
            st.success("Résultats PMU mis à jour !")
            st.rerun()
        else:
            st.info("Aucun résultat supplémentaire trouvé.")

    historique = charger_historique()
    paris_regles = [p for p in historique if p.get("statut") in ["Gagné", "Perdu"]]
    total_mises = sum(safe_float(p.get("mise", 0)) for p in paris_regles)
    total_gains = sum(safe_float(p.get("gain", 0)) for p in paris_regles if p.get("statut") == "Gagné")
    bilan_net = total_gains - total_mises
    roi = (bilan_net / total_mises * 100) if total_mises > 0 else 0.0
    taux_win = (sum(1 for p in paris_regles if p.get("statut") == "Gagné") / len(paris_regles) * 100) if paris_regles else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Mises", f"{total_mises:.2f} €")
    kpi2.metric("Total Gains", f"{total_gains:.2f} €")
    kpi3.metric("Bilan Net", f"{bilan_net:+.2f} €")
    kpi4.metric("ROI / Taux Réussite", f"{roi:+.1f}% | Win: {taux_win:.1f}%")

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
    st.subheader("📋 Historique Détaillé des Paris")
    if historique:
        st.dataframe(pd.DataFrame(historique), width="stretch")
    else:
        st.info("Aucun pari enregistré.")

# ================= 6. BILAN PAR RÉUNION =================
with tab_bilan:
    st.subheader("🏟️ Bilan Financier par Réunion & Discipline")
    st.subheader("📊 Performance SQL par Discipline")
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
        st.info("Aucun pari réglé (Gagné/Perdu) disponible pour afficher le bilan journalier.")

# ================= 7. ADMINISTRATION =================
with tab_admin:
    st.subheader("🛠️ Panneau d'Administration Avancé")
    mdp_admin = st.text_input("Code Admin requis", type="password", key="input_mdp_admin_secu")
    
    if mdp_admin.strip() == st.secrets.get("PASSWORD", "301180"):
        st.success("Accès administrateur autorisé.")
        
        st.divider()
        st.subheader("🤖 Automatisation des Paris sur une Période")
        st.markdown("Sélectionne une plage de dates pour télécharger, analyser et enregistrer automatiquement les paris sur toutes les courses de la période.")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            date_debut = st.date_input("Date de début", key="admin_date_debut")
        with col_d2:
            date_fin = st.date_input("Date de fin", key="admin_date_fin")
            
        mise_auto_defaut = st.number_input("Mise par course par défaut (€)", min_value=1, value=10, key="admin_mise_auto")
        
        if st.button("🚀 Lancer l'automatisation des paris sur la période", key="btn_lancer_auto_periode"):
            if date_debut > date_fin:
                st.error("La date de début doit être antérieure ou égale à la date de fin.")
            else:
                delta = date_fin - date_debut
                nb_jours = delta.days + 1
                
                total_paris_ajoutes = 0
                historique_actuel = charger_historique()
                
                for i in range(nb_jours):
                    courante_dt = date_debut + datetime.timedelta(days=i)
                    courante_iso = courante_dt.strftime("%Y-%m-%d")
                    
                    st.markdown(f"### 📅 Traitement du : {courante_iso}")
                    
                    # 1. Téléchargement via la fonction officielle (qui gère son propre affichage et stockage)
                    succes_dl = telecharger_pmu_date(courante_iso, None)
                    
                    if not succes_dl:
                        st.warning(f"Impossible de récupérer les données pour le {courante_iso} (pas de courses ou date trop ancienne).")
                        continue
                        
                    # 2. Charger les courses du jour fraîchement téléchargées
                    donnees_jour, _ = charger_courses_jour_db(courante_iso)
                    if not donnees_jour:
                        continue
                        
                    nb_courses_jour = 0
                    for c_elem in donnees_jour:
                        nom_c = str(c_elem.get("nom_course", "")).strip()
                        if not nom_c or nom_c.isdigit() or len(nom_c) <= 2:
                            continue
                            
                        r_nom = f"{c_elem.get('reunion', 'R1')} - {c_elem.get('hippodrome', 'HIPPODROME')}"
                        chevaux_c = c_elem.get("chevaux", [])
                        chevaux_val_c = [c for c in chevaux_c if safe_float(c.get("cote")) > 1.0 or c.get("cote") is None]
                        
                        if not chevaux_val_c:
                            continue
                            
                        # 3. Application de l'analyse IA / Value Bet
                        params_ad_chrono = calculer_parametres_adaptatifs()
                        for c in chevaux_val_c:
                            c["score_analyse"] = evaluer_score_cheval(
                                c, c_elem.get("discipline"), c_elem.get("terrain_officiel"),
                                c_elem.get("corde", "Corde standard"), courante_iso, params_ad_chrono, distance_course=c_elem.get("distance", "")
                            )
                        normaliser_scores_chevaux(chevaux_val_c, "score_analyse")
                        calculer_valeur_esperee_avancee(chevaux_val_c, len(chevaux_c))
                        
                        chevaux_val_c.sort(key=lambda x: (x.get("ev_index", 0), x["score_analyse"]), reverse=True)
                        b_chev = chevaux_val_c[0]
                        outsiders_c = [c for c in chevaux_val_c if 7.0 <= safe_float(c.get("cote")) <= 22.0 and c["num"] != b_chev["num"]]
                        p_chev = max(outsiders_c, key=lambda x: x.get("ev_index", 0)) if outsiders_c else (chevaux_val_c[1] if len(chevaux_val_c) > 1 else b_chev)
                        
                        mise_input = float(mise_auto_defaut)
                        mise_secu = round(mise_input * 0.8, 1)
                        mise_poker = round(mise_input - mise_secu, 1)
                        
                        nouveau_pari = {
                            "date": courante_iso,
                            "reunion": r_nom,
                            "course_num": c_elem.get("course", "C1"),
                            "course": f"{r_nom} - {c_elem.get('course', 'C1')}",
                            "discipline": c_elem.get("discipline"),
                            "type": "Rapide Value (Auto)",
                            "details": f"Simple Placé (Sécurité Value) ➔ N°{b_chev.get('num', '?')} ({mise_secu}€) | Simple Gagnant (Poker Value) ➔ N°{p_chev.get('num', '?')} ({mise_poker}€)",
                            "mise": mise_input,
                            "statut": "En attente",
                            "gain": 0.0,
                            "diagnostic": "",
                        }
                        historique_actuel.append(nouveau_pari)
                        total_paris_ajoutes += 1
                        nb_courses_jour += 1
                        
                    st.success(f"-> {nb_courses_jour} paris générés pour le {courante_iso}.")
                    
                sauvegarder_historique(historique_actuel)
                st.balloons()
                st.success(f"🎉 Automatisation globale terminée avec succès ! {total_paris_ajoutes} paris au total ont été enregistrés.")

        st.divider()
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("🧹 Vider le cache complet Streamlit", key="btn_vider_cache_admin"):
                st.cache_data.clear()
                st.success("Cache vidé !")
        with col_a2:
            if st.button("🔄 Forcer la migration Base de Données", key="btn_migrer_db_admin"):
                init_db()
                migrer_anciens_json_vers_sqlite()
                st.success("Migration et initialisation de la base de données effectuées.")

        # --- NOUVELLE ZONE : REMISE À ZÉRO COMPLÈTE ---
        st.divider()
        st.subheader("⚠️ Zone Dangereuse : Remise à Zéro Complète")
        st.markdown("Cette action supprimera définitivement **tous les paris enregistrés, les courses en cache et l'historique** de la base de données.")
        
        confirm_raz = st.checkbox("Je confirme vouloir supprimer TOUTES les données (irréversible)", key="confirm_raz_checkbox")
        if confirm_raz:
            if st.button("🗑️ Supprimer TOUTES les données acquises", key="btn_raz_total", type="primary"):
                try:
                    # Connexion à la base et suppression de toutes les lignes de chaque table (évite le blocage Windows)
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    for table_name in tables:
                        t = table_name[0]
                        if t != 'sqlite_sequence':
                            cursor.execute(f"DELETE FROM {t};")
                    conn.commit()
                    conn.close()
                    
                    # Réinitialisation et vidage du cache Streamlit
                    init_db()
                    st.cache_data.clear()
                    
                    st.success("🗑️ Remise à zéro effectuée avec succès ! Toutes les données ont été supprimées.")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la remise à zéro : {e}")
