import json
import re
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import psycopg2
import streamlit as st


def safe_float(val, default=0.0):
    if val is None or val == "" or val == "-":
        return default
    try:
        return float(str(val).replace(",", ".").replace("€", "").strip())
    except (ValueError, TypeError):
        return default


def get_conn():
    try:
        cfg = st.secrets["supabase"]
    except Exception:
        cfg = {
            "host": "localhost",
            "database": "postgres",
            "user": "postgres",
            "password": "postgres",
            "port": 5432,
        }

    return psycopg2.connect(
        host=cfg["host"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        port=cfg["port"],
    )


@contextmanager
def get_db_connection():
    """Ouvre une connexion directe unique et la ferme proprement."""
    conn = get_conn()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


DB_PATH = "supabase_cloud"


@st.cache_resource
def init_db():
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pmu_bets (
                    id SERIAL PRIMARY KEY,
                    date TEXT,
                    reunion TEXT,
                    course TEXT,
                    cheval TEXT,
                    cote REAL,
                    mise REAL,
                    statut TEXT,
                    gain REAL,
                    discipline TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS pmu_bankroll (
                    id SERIAL PRIMARY KEY,
                    montant REAL,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS pmu_predictions_cache (
                    course_id TEXT PRIMARY KEY,
                    prediction_data TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS modele_ia (
                    cle TEXT PRIMARY KEY,
                    valeur TEXT
                );
                CREATE TABLE IF NOT EXISTS courses_cache (
                    date_iso TEXT PRIMARY KEY,
                    data_json TEXT
                );
                """
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur initialisation de la base de données Supabase : {e}")
            return False


# Compatibilité imports
def migrer_anciens_json_vers_sqlite(*args, **kwargs):
    return True


def get_connection():
    return get_conn()


# ==========================================
# GESTION DES PARIS / HISTORIQUE / ROI
# ==========================================

def save_bet_to_db(
    date,
    reunion,
    course,
    cheval,
    cote,
    mise,
    statut="En cours",
    gain=0.0,
    discipline="Non renseigné",
):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pmu_bets (date, reunion, course, cheval, cote, mise, statut, gain, discipline, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(date),
                    str(reunion),
                    str(course),
                    str(cheval),
                    float(cote),
                    float(mise),
                    str(statut),
                    float(gain),
                    str(discipline),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur lors de l'enregistrement du pari sur Supabase : {e}")
            return None


def save_multiple_bets_to_db(liste_paris):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            query = """
                INSERT INTO pmu_bets (date, reunion, course, cheval, cote, mise, statut, gain, discipline, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            valeurs = [
                (
                    str(p["date"]),
                    str(p["reunion"]),
                    str(p["course"]),
                    str(p["cheval"]),
                    float(p["cote"]),
                    float(p["mise"]),
                    str(p.get("statut", "En attente")),
                    float(p.get("gain", 0.0)),
                    str(p.get("discipline", "Non renseigné")),
                    datetime.utcnow().isoformat(),
                )
                for p in liste_paris
            ]
            cur.executemany(query, valeurs)
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur lors de l'enregistrement groupé des paris sur Supabase : {e}")
            return False


def load_bets_from_db():
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM pmu_bets ORDER BY id")
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description] if cur.description else []
            cur.close()
            if rows and col_names:
                df = pd.DataFrame(rows, columns=col_names)
                if "mise" in df.columns:
                    df["mise"] = pd.to_numeric(df["mise"], errors="coerce").fillna(0.0)
                if "gain" in df.columns:
                    df["gain"] = pd.to_numeric(df["gain"], errors="coerce").fillna(0.0)
                if "cote" in df.columns:
                    df["cote"] = pd.to_numeric(df["cote"], errors="coerce").fillna(0.0)
                return df
            return pd.DataFrame(
                columns=[
                    "id",
                    "date",
                    "reunion",
                    "course",
                    "cheval",
                    "cote",
                    "mise",
                    "statut",
                    "gain",
                    "discipline",
                    "created_at",
                ]
            )
        except Exception:
            return pd.DataFrame(
                columns=[
                    "id",
                    "date",
                    "reunion",
                    "course",
                    "cheval",
                    "cote",
                    "mise",
                    "statut",
                    "gain",
                    "discipline",
                    "created_at",
                ]
            )


def update_bet_status(bet_id, statut, gain):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE pmu_bets SET statut = %s, gain = %s WHERE id = %s
                """,
                (str(statut), float(gain), bet_id),
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur lors de la mise à jour du pari : {e}")
            return False


def charger_historique():
    df = load_bets_from_db()
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    for rec in records:
        if "cheval" in rec and "details" not in rec:
            rec["details"] = rec["cheval"]
        if "course" in rec and "course_num" not in rec:
            rec["course_num"] = rec["course"]
    return records


def sauvegarder_historique(hist):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            for pari_item in hist:
                date = str(pari_item.get("date", ""))
                reunion = str(pari_item.get("reunion", ""))
                course = str(pari_item.get("course_num", pari_item.get("course", "")))
                cheval = str(pari_item.get("details", pari_item.get("cheval", "")))
                statut = str(pari_item.get("statut", "En attente"))
                gain = float(pari_item.get("gain", 0.0))
                mise = float(pari_item.get("mise", 0.0))
                cote = float(pari_item.get("cote", 0.0))
                discipline = str(pari_item.get("discipline", "Non renseigné"))

                cur.execute(
                    """
                    SELECT id FROM pmu_bets
                    WHERE date = %s AND reunion = %s AND course = %s AND cheval = %s
                    """,
                    (date, reunion, course, cheval),
                )
                row = cur.fetchone()

                if row:
                    bet_id = row[0]
                    cur.execute(
                        """
                        UPDATE pmu_bets
                        SET statut = %s, gain = %s, discipline = %s
                        WHERE id = %s
                        """,
                        (statut, gain, discipline, bet_id),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO pmu_bets (date, reunion, course, cheval, cote, mise, statut, gain, discipline, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            date,
                            reunion,
                            course,
                            cheval,
                            cote,
                            mise,
                            statut,
                            gain,
                            discipline,
                            datetime.utcnow().isoformat(),
                        ),
                    )

            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde de l'historique sur Supabase : {e}")
            return False


# ==========================================
# GESTION DU BANKROLL
# ==========================================

def get_bankroll():
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT montant FROM pmu_bankroll ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            cur.close()
            if row:
                return float(row[0])
            return 1000.0
        except Exception:
            return 1000.0


def update_bankroll(nouveau_montant):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pmu_bankroll (montant, created_at) VALUES (%s, %s)
                """,
                (float(nouveau_montant), datetime.utcnow().isoformat()),
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur lors de la mise à jour du bankroll : {e}")
            return False


# ==========================================
# CACHE / PRÉDICTIONS & COURSES
# ==========================================

def save_prediction_cache(course_id, prediction_json):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pmu_predictions_cache (course_id, prediction_data, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (course_id) DO UPDATE SET prediction_data = EXCLUDED.prediction_data, created_at = EXCLUDED.created_at
                """,
                (str(course_id), json.dumps(prediction_json), datetime.utcnow().isoformat()),
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur sauvegarde cache de prédiction : {e}")
            return False


def get_prediction_cache(course_id):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT prediction_data FROM pmu_predictions_cache WHERE course_id = %s",
                (str(course_id),),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                data = row[0]
                return json.loads(data) if isinstance(data, str) else data
            return None
        except Exception:
            return None


def charger_courses_jour_db(date_iso):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT data_json FROM courses_cache WHERE date_iso = %s",
                (str(date_iso),),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                data_str = row[0]
                data = json.loads(data_str) if isinstance(data_str, str) else data_str
                reunions_map = {}
                for c in data:
                    r = c.get("reunion", "R1")
                    if r not in reunions_map:
                        reunions_map[r] = []
                    reunions_map[r].append(c)
                return data, reunions_map
        except Exception as e:
            st.error(f"Erreur chargement courses depuis Supabase : {e}")
        return [], {}


def sauvegarder_courses_jour_db(date_iso, data):
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            data_json_str = json.dumps(data, ensure_ascii=False)
            cur.execute(
                """
                INSERT INTO courses_cache (date_iso, data_json)
                VALUES (%s, %s)
                ON CONFLICT (date_iso) DO UPDATE SET data_json = EXCLUDED.data_json
                """,
                (str(date_iso), data_json_str),
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            st.error(f"Erreur sauvegarde courses sur Supabase : {e}")
            return False


# ==========================================
# ANALYSES ET RAPPORTS
# ==========================================

def extraire_cote_depuis_details(row):
    """Extrait la cote enregistrée dans le champ 'cote' ou dans le texte 'cheval'."""
    cote_val = row.get("cote", 0.0)
    if pd.notnull(cote_val) and float(cote_val) > 1.0:
        return float(cote_val)

    details = str(row.get("cheval", ""))
    match = re.search(r"Cote:\s*(\d+(?:[\.,]\d+)?)", details)
    if match:
        return float(match.group(1).replace(",", "."))
    return 0.0


def analyser_roi_par_discipline_sql():
    df = load_bets_from_db()

    cotes_podium_par_discipline = {}
    total_podiums_disc = {}

    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT data_json FROM courses_cache")
            rows = cur.fetchall()
            cur.close()

            for row in rows:
                val_json = row[0] if isinstance(row, (list, tuple)) else row.get("data_json")
                try:
                    races = json.loads(val_json)
                    if isinstance(races, dict):
                        races = [races]
                    if isinstance(races, list):
                        for race in races:
                            disc = str(race.get("discipline", "Non renseigné")).strip()
                            chevaux = race.get("chevaux", [])
                            for part in chevaux:
                                ordre = part.get("ordreArrivee")
                                if isinstance(ordre, int) and 1 <= ordre <= 3:
                                    cote_cheval = safe_float(part.get("cote", 0.0))
                                    if cote_cheval > 1.0:
                                        if disc not in cotes_podium_par_discipline:
                                            cotes_podium_par_discipline[disc] = 0.0
                                            total_podiums_disc[disc] = 0
                                        cotes_podium_par_discipline[disc] += cote_cheval
                                        total_podiums_disc[disc] += 1
                except Exception:
                    continue
        except Exception as e:
            st.error(f"Erreur lecture cache courses pour le top 3 : {e}")

    moy_cotes_podium = {}
    for disc, somme in cotes_podium_par_discipline.items():
        nb = total_podiums_disc.get(disc, 1)
        moy_cotes_podium[disc] = round(somme / nb, 2) if nb > 0 else 0.0

    if df.empty or "discipline" not in df.columns:
        data_fallback = []
        for disc, moy in moy_cotes_podium.items():
            data_fallback.append(
                {
                    "Discipline": disc,
                    "Nb Paris": 0,
                    "Total Mises (€)": 0.0,
                    "Total Gains (€)": 0.0,
                    "Bilan Net (€)": 0.0,
                    "ROI (%)": 0.0,
                    "Moy. Cotes Réelles Top 3": moy,
                }
            )
        return pd.DataFrame(data_fallback)

    df["discipline"] = df["discipline"].fillna("Non renseigné").replace("", "Non renseigné")
    df_regle = df[df["statut"].isin(["Gagné", "Perdu"])].copy()

    if df_regle.empty:
        grouped = df.groupby("discipline").agg(
            nb_paris=("id", "count"),
            total_mises=("mise", lambda x: 0.0),
            total_gains=("gain", lambda x: 0.0),
        ).reset_index()
    else:
        df_regle["mise"] = pd.to_numeric(df_regle["mise"], errors="coerce").fillna(0.0)
        df_regle["gain"] = pd.to_numeric(df_regle["gain"], errors="coerce").fillna(0.0)

        grouped = df_regle.groupby("discipline").agg(
            nb_paris=("id", "count"),
            total_mises=("mise", "sum"),
            total_gains=("gain", lambda x: df_regle.loc[x.index, "gain"][df_regle.loc[x.index, "statut"] == "Gagné"].sum()),
        ).reset_index()

    grouped["bilan_net"] = grouped["total_gains"] - grouped["total_mises"]
    grouped["roi_pourcent"] = grouped.apply(
        lambda row: (row["bilan_net"] / row["total_mises"] * 100) if row["total_mises"] > 0 else 0.0,
        axis=1,
    )

    grouped["Moy. Cotes Réelles Top 3"] = grouped["discipline"].map(lambda d: moy_cotes_podium.get(d, 0.0))

    grouped = grouped.sort_values(by="roi_pourcent", ascending=False)
    grouped.columns = [
        "Discipline",
        "Nb Paris",
        "Total Mises (€)",
        "Total Gains (€)",
        "Bilan Net (€)",
        "ROI (%)",
        "Moy. Cotes Réelles Top 3",
    ]
    return grouped


def analyser_bilan_journalier_sql():
    df = load_bets_from_db()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df_regle = df[df["statut"].isin(["Gagné", "Perdu"])].copy()
    if df_regle.empty:
        return pd.DataFrame()

    df_regle["mise"] = pd.to_numeric(df_regle["mise"], errors="coerce").fillna(0.0)
    df_regle["gain"] = pd.to_numeric(df_regle["gain"], errors="coerce").fillna(0.0)

    grouped = df_regle.groupby("date").agg(
        nb_paris=("id", "count"),
        total_mises=("mise", "sum"),
        total_gains=("gain", lambda x: df_regle.loc[x.index, "gain"][df_regle.loc[x.index, "statut"] == "Gagné"].sum()),
    ).reset_index()

    grouped["bilan_net"] = grouped["total_gains"] - grouped["total_mises"]
    grouped["roi_pourcent"] = grouped.apply(
        lambda row: (row["bilan_net"] / row["total_mises"] * 100) if row["total_mises"] > 0 else 0.0,
        axis=1,
    )
    grouped = grouped.sort_values(by="date", ascending=False)
    grouped.columns = ["Date", "Nb Paris", "Total Mises (€)", "Total Gains (€)", "Bilan Net (€)", "ROI (%)"]
    return grouped


__all__ = [
    "safe_float",
    "get_conn",
    "get_db_connection",
    "DB_PATH",
    "init_db",
    "migrer_anciens_json_vers_sqlite",
    "get_connection",
    "save_bet_to_db",
    "save_multiple_bets_to_db",
    "load_bets_from_db",
    "update_bet_status",
    "charger_historique",
    "sauvegarder_historique",
    "get_bankroll",
    "update_bankroll",
    "save_prediction_cache",
    "get_prediction_cache",
    "charger_courses_jour_db",
    "sauvegarder_courses_jour_db",
    "extraire_cote_depuis_details",
    "analyser_roi_par_discipline_sql",
    "analyser_bilan_journalier_sql",
]