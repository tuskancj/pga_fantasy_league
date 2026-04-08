import json
import os
import traceback
import boto3
import requests
import numpy as np
import pandas as pd
import psycopg2

from datetime import datetime


# ---------------------------------------------------------------------------
# Secrets / config
# ---------------------------------------------------------------------------

def get_secret(secret_name: str) -> dict:
    """Pull a JSON secret from AWS Secrets Manager."""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def load_config() -> dict:
    """
    Expects two environment variables set on the Lambda:
      SECRET_NAME  – name of the Secrets Manager secret containing DB creds
                     (keys: host, port, dbname, username, password, api_key)

    Alternatively every value can be its own env-var if you prefer.
    """
    secret_name = os.environ.get("SECRET_NAME")
    if secret_secret_name := secret_name:
        secrets = get_secret(secret_secret_name)
        return {
            "api_key":  secrets["api_key"],
            "host":     secrets["host"],
            "port":     secrets.get("port", "5432"),
            "dbname":   secrets.get("dbname", "postgres"),
            "user":     secrets["username"],
            "password": secrets["password"],
        }

    # Fallback: individual environment variables (handy for local testing)
    return {
        "api_key":  os.environ["DG_API_KEY"],
        "host":     os.environ["DB_HOST"],
        "port":     os.environ.get("DB_PORT", "5432"),
        "dbname":   os.environ.get("DB_NAME", "postgres"),
        "user":     os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_connection(cfg: dict):
    return psycopg2.connect(
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
    )


def read_sql(sql: str, cfg: dict) -> pd.DataFrame:
    conn = get_connection(cfg)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def insert_df(table_name: str, df: pd.DataFrame, cfg: dict) -> None:
    conn = get_connection(cfg)
    try:
        cursor = conn.cursor()
        cols = ", ".join(df.columns)
        placeholders = ", ".join(["%s"] * len(df.columns))
        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        data = df.to_records(index=False).tolist()
        cursor.executemany(sql, data)
        conn.commit()
        print(f"[insert_df] Inserted {len(data)} rows into '{table_name}'.")
    except psycopg2.Error as e:
        conn.rollback()
        raise RuntimeError(f"Insert into '{table_name}' failed: {e.pgerror}") from e
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# DataGolf API helpers
# ---------------------------------------------------------------------------

def dg_get(endpoint: str, params: dict) -> pd.DataFrame:
    """GET a DataGolf JSON feed and return a DataFrame."""
    url = f"https://feeds.datagolf.com/{endpoint}"
    params["file_format"] = "json"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def dg_raw(endpoint: str, params: dict) -> requests.Response:
    """GET a DataGolf feed and return the raw Response (for status checks)."""
    url = f"https://feeds.datagolf.com/{endpoint}"
    params["file_format"] = "json"
    return requests.get(url, params=params, timeout=30)


# ---------------------------------------------------------------------------
# ETL steps
# ---------------------------------------------------------------------------

def sync_players(cfg: dict, api_key: str) -> None:
    print("[sync_players] Checking for missing players...")
    players_dg = dg_get("get-player-list", {"key": api_key})
    players_aws = read_sql("SELECT DISTINCT dg_id FROM player;", cfg)

    missing = players_dg[~players_dg["dg_id"].isin(players_aws["dg_id"])]
    if not missing.empty:
        print(f"[sync_players] Adding {len(missing)} missing player(s).")
        insert_df("player", missing, cfg)
    else:
        print("[sync_players] No missing players.")


def get_missing_events(cfg: dict, api_key: str, current_year: int) -> pd.DataFrame:
    """Return events that exist in DataGolf but not in the AWS DB (current year, PGA only)."""
    print("[get_missing_events] Checking for missing events...")
    events_dg = dg_get("historical-raw-data/event-list", {"key": api_key})
    events_aws = read_sql(
        f"SELECT event_id FROM event WHERE calendar_year = {current_year};", cfg
    )

    events_dg_year = events_dg[
        (events_dg["tour"] == "pga")
        & (events_dg["calendar_year"] == events_dg["calendar_year"].max())
    ]

    missing = events_dg_year[~events_dg_year["event_id"].isin(events_aws["event_id"])]
    print(f"[get_missing_events] Found {len(missing)} missing event(s).")
    return missing


def filter_events_with_dfs(
    missing_events: pd.DataFrame, api_key: str, current_year: int
) -> pd.DataFrame:
    """Remove events that don't yet have DFS data (API returns 400)."""
    print("[filter_events_with_dfs] Checking DFS availability...")
    no_dfs = []
    for ev_id in missing_events["event_id"]:
        resp = dg_raw(
            "historical-dfs-data/points",
            {"tour": "pga", "site": "fanduel", "event_id": ev_id, "year": current_year, "key": api_key},
        )
        if resp.status_code == 400:
            no_dfs.append(ev_id)

    filtered = missing_events[~missing_events["event_id"].isin(no_dfs)]
    print(f"[filter_events_with_dfs] {len(filtered)} event(s) have DFS data ready.")
    return filtered


def sync_events(missing_events: pd.DataFrame, cfg: dict) -> None:
    if missing_events.empty:
        return
    to_insert = (
        missing_events
        .copy()
        .drop(["sg_categories", "tour", "traditional_stats"], axis=1)
    )
    to_insert["dfs_payout"] = None
    insert_df("event", to_insert, cfg)


def sync_rounds(
    missing_events: pd.DataFrame, cfg: dict, api_key: str, current_year: int
) -> None:
    if missing_events.empty:
        print("[sync_rounds] No missing events — skipping rounds sync.")
        return

    print("[sync_rounds] Fetching round data...")

    # Use a well-documented event to derive the canonical column list
    ref = dg_get(
        "historical-raw-data/rounds",
        {"tour": "pga", "event_id": 2, "year": 2025, "key": api_key},
    )
    basic_names = ["event_id", "calendar_year", "dg_id", "round", "fin_text"]
    stat_names = basic_names + (
        pd.json_normalize(ref["scores"].loc[0]["round_1"], max_level=0)
        .columns.tolist()
    )
    stat_names = sorted(set(stat_names))

    rounds = []

    for ev_id in missing_events["event_id"]:
        event_data = dg_get(
            "historical-raw-data/rounds",
            {"tour": "pga", "event_id": ev_id, "year": current_year, "key": api_key},
        )
        player_summary = pd.json_normalize(event_data["scores"], max_level=0)

        for _, player in player_summary.iterrows():
            event_rounds = []
            for rnd_key in ["round_1", "round_2", "round_3", "round_4"]:
                if pd.notna(player.get(rnd_key)):
                    event_rounds.append(pd.json_normalize(player[rnd_key], max_level=0))

            for idx, rnd in enumerate(event_rounds):
                basics = pd.DataFrame({
                    "dg_id":         [player["dg_id"]],
                    "fin_text":      [player["fin_text"]],
                    "round":         [idx + 1],
                    "event_id":      [ev_id],
                    "calendar_year": [current_year],
                })
                for col in basics.columns:
                    rnd[col] = basics[col].values

                for missing_col in set(stat_names) - set(rnd.columns):
                    rnd[missing_col] = None

                rnd_sorted = rnd[sorted(rnd.columns)]
                rounds.append(rnd_sorted.iloc[0].values)

    if not rounds:
        print("[sync_rounds] No round rows to insert.")
        return

    rounds_df = (
        pd.DataFrame(rounds, columns=stat_names)
        .replace([np.nan, "missing"], None)
    )

    # Sync missing courses first
    _sync_courses(rounds_df, cfg)

    # Map event_id / course_num → AWS UUIDs
    events_aws = read_sql(
        f"SELECT id_event, event_id, calendar_year FROM event WHERE calendar_year = {current_year};",
        cfg,
    )
    events_aws["id_event"] = events_aws["id_event"].astype(str)
    courses_aws = read_sql("SELECT id_course, course_num FROM course;", cfg)

    rounds_clean = rounds_df.drop(["course_name", "course_par", "fin_text"], axis=1)
    rounds_clean = (
        rounds_clean
        .merge(events_aws, on=["event_id", "calendar_year"], how="left")
        .drop(["event_id", "calendar_year"], axis=1)
        .merge(courses_aws, on="course_num", how="left")
        .drop("course_num", axis=1)
    )

    print(f"[sync_rounds] Inserting {len(rounds_clean)} round rows.")
    insert_df("round", rounds_clean, cfg)


def _sync_courses(rounds_df: pd.DataFrame, cfg: dict) -> None:
    new_courses = (
        rounds_df
        .drop_duplicates("course_num")[["course_name", "course_num", "course_par"]]
        .reset_index(drop=True)
    )
    courses_aws = read_sql("SELECT * FROM course;", cfg)
    missing = new_courses[~new_courses["course_num"].isin(courses_aws["course_num"])]
    if not missing.empty:
        print(f"[_sync_courses] Adding {len(missing)} missing course(s).")
        insert_df("course", missing, cfg)
    else:
        print("[_sync_courses] No missing courses.")


def sync_dfs(
    missing_events: pd.DataFrame, cfg: dict, api_key: str, current_year: int
) -> None:
    if missing_events.empty:
        print("[sync_dfs] No missing events — skipping DFS sync.")
        return

    print("[sync_dfs] Fetching DFS data...")

    # Canonical column list from a known-complete event
    ref = dg_get(
        "historical-dfs-data/points",
        {"tour": "pga", "site": "fanduel", "event_id": 2, "year": 2025, "key": api_key},
    )
    stat_names = sorted(
        pd.json_normalize(ref["dfs_points"].loc[0], max_level=0)
        .drop(["ownership", "player_name"], axis=1)
        .columns.tolist()
        + ["id_event"]
    )

    events_aws = read_sql(
        f"SELECT id_event, event_id, calendar_year FROM event WHERE calendar_year = {current_year};",
        cfg,
    )
    events_aws["id_event"] = events_aws["id_event"].astype(str)

    dfs_rows = []

    for ev_id in missing_events["event_id"]:
        event_data = dg_get(
            "historical-dfs-data/points",
            {"tour": "pga", "site": "fanduel", "event_id": ev_id, "year": current_year, "key": api_key},
        )

        for player in event_data["dfs_points"]:
            dfs_pts = (
                pd.json_normalize(player, max_level=0)
                .drop(["player_name", "ownership"], axis=1)
            )
            dfs_pts["calendar_year"] = current_year
            dfs_pts["event_id"] = ev_id
            dfs_pts = (
                dfs_pts
                .merge(events_aws, on=["event_id", "calendar_year"], how="left")
                .drop(["calendar_year", "event_id"], axis=1)
            )

            for missing_col in set(stat_names) - set(dfs_pts.columns):
                dfs_pts[missing_col] = None

            dfs_pts_sorted = dfs_pts[sorted(dfs_pts.columns)]
            dfs_rows.append(dfs_pts_sorted.iloc[0].values)

    if not dfs_rows:
        print("[sync_dfs] No DFS rows to insert.")
        return

    dfs_df = (
        pd.DataFrame(dfs_rows, columns=stat_names)
        .replace([np.nan, "missing"], None)
    )

    print(f"[sync_dfs] Inserting {len(dfs_df)} DFS rows.")
    insert_df("dfs_total", dfs_df, cfg)


# ---------------------------------------------------------------------------
# SNS email helper
# ---------------------------------------------------------------------------

def send_email(subject: str, body: str) -> None:
    """
    Publish a message to the SNS topic defined by the SNS_TOPIC_ARN env var.
    Silently skips if the env var is not set (e.g. local testing).
    """
    topic_arn = os.environ.get("SNS_TOPIC_ARN")
    if not topic_arn:
        print("[send_email] SNS_TOPIC_ARN not set — skipping email.")
        return
    try:
        boto3.client("sns").publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=body,
        )
        print(f"[send_email] Email sent: {subject}")
    except Exception as e:
        # Never let a notification failure crash the function
        print(f"[send_email] Failed to send email: {e}")


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    log: list[str] = []   # collects step-by-step results for the email body

    def step(msg: str) -> None:
        print(msg)
        log.append(msg)

    try:
        cfg = load_config()
        api_key = cfg.pop("api_key")
        current_year = datetime.now().year

        step(f"PGA DB Sync started at {run_time}")
        step("-" * 50)

        # Players
        players_dg  = dg_get("get-player-list", {"key": api_key})
        players_aws = read_sql("SELECT DISTINCT dg_id FROM player;", cfg)
        missing_players = players_dg[~players_dg["dg_id"].isin(players_aws["dg_id"])]
        if not missing_players.empty:
            insert_df("player", missing_players, cfg)
            step(f"Players added: {len(missing_players)}")
        else:
            step("Players: up to date")

        # Events
        missing_events = get_missing_events(cfg, api_key, current_year)

        if missing_events.empty:
            step("Events: no new events found — nothing else to do.")
            body = "\n".join(log)
            send_email(f"PGA DB Sync — No New Events ({run_time})", body)
            return {"statusCode": 200, "body": "No new events."}

        missing_events = filter_events_with_dfs(missing_events, api_key, current_year)

        if missing_events.empty:
            step("Events: found new events but DFS data not ready yet.")
            body = "\n".join(log)
            send_email(f"PGA DB Sync — DFS Not Ready Yet ({run_time})", body)
            return {"statusCode": 200, "body": "No DFS-ready events."}

        event_names = missing_events["event_name"].tolist()
        step(f"Events to process ({len(event_names)}): {', '.join(event_names)}")

        # Sync each table
        sync_events(missing_events, cfg)
        step(f"Events inserted: {len(missing_events)}")

        sync_rounds(missing_events, cfg, api_key, current_year)
        step("Rounds: inserted successfully")

        sync_dfs(missing_events, cfg, api_key, current_year)
        step("DFS scores: inserted successfully")

        step("-" * 50)
        step("All steps completed successfully.")

        send_email(
            subject=f"PGA DB Sync — Success ({run_time})",
            body="\n".join(log),
        )
        return {"statusCode": 200, "body": f"Processed {len(missing_events)} event(s)."}

    except Exception as exc:
        tb = traceback.format_exc()
        log.append("-" * 50)
        log.append(f"ERROR: {exc}")
        log.append("")
        log.append("Full traceback:")
        log.append(tb)

        print(f"[lambda_handler] ERROR: {exc}\n{tb}")

        send_email(
            subject=f"🚨 ERROR! PGA DB Sync Failed ({run_time})",
            body="\n".join(log),
        )
        raise   # re-raise so Lambda marks the invocation as failed
