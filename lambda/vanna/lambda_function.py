import json
import os
import boto3

from vanna.legacy.anthropic.anthropic_chat import Anthropic_Chat
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore

TRAINING_EXAMPLES = [
    {"question": "What are cjt3's most picked golfers?", "sql": "SELECT SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer, COUNT(*) AS times_picked FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN player p ON p.dg_id=dt.dg_id WHERE a.username='cjt3' GROUP BY p.player_name ORDER BY times_picked DESC LIMIT 20;"},
    {"question": "Who won the most events in 2026?", "sql": "WITH scores AS (SELECT a.username, e.id_event, SUM(dt.total_pts) AS pts FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY a.username,e.id_event), ranked AS (SELECT *, RANK() OVER (PARTITION BY id_event ORDER BY pts DESC) AS rnk FROM scores) SELECT username, COUNT(*) AS wins FROM ranked WHERE rnk=1 GROUP BY username ORDER BY wins DESC;"},
    {"question": "How did Scheffler perform at the Masters across all years?", "sql": "SELECT e.calendar_year, e.event_name, ROUND(dt.total_pts::numeric,1) AS pts, dt.fin_text, dt.salary FROM dfs_total dt JOIN player p ON p.dg_id=dt.dg_id JOIN event e ON e.id_event=dt.id_event WHERE p.player_name LIKE '%Scheffler%' AND e.event_name LIKE '%Masters%' ORDER BY e.calendar_year DESC LIMIT 100;"},
    {"question": "What are the current season standings?", "sql": "SELECT a.username, ROUND(SUM(dt.total_pts)::numeric,1) AS total_pts, COUNT(DISTINCT e.id_event) AS events FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 AND a.active=true GROUP BY a.username ORDER BY total_pts DESC;"},
    {"question": "Which golfers should I pick for the Masters based on history?", "sql": "SELECT SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer, COUNT(*) AS appearances, ROUND(AVG(dt.total_pts)::numeric,1) AS avg_pts, ROUND(MAX(dt.total_pts)::numeric,1) AS best_pts, ROUND(AVG(dt.salary)::numeric,0) AS avg_salary FROM dfs_total dt JOIN player p ON p.dg_id=dt.dg_id JOIN event e ON e.id_event=dt.id_event WHERE e.event_name LIKE '%Masters%' AND dt.fin_text NOT IN ('WD','CUT') GROUP BY p.player_name ORDER BY appearances DESC, avg_pts DESC LIMIT 20;"},
    {"question": "What is the salary vs points value for all golfers at the Masters?", "sql": "SELECT SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer, ROUND(AVG(dt.salary)::numeric,0) AS avg_salary, ROUND(AVG(dt.total_pts)::numeric,1) AS avg_pts, ROUND((AVG(dt.total_pts)/NULLIF(AVG(dt.salary),0)*1000)::numeric,2) AS value_ratio FROM dfs_total dt JOIN player p ON p.dg_id=dt.dg_id JOIN event e ON e.id_event=dt.id_event WHERE e.event_name LIKE '%Masters%' AND dt.salary>0 GROUP BY p.player_name ORDER BY avg_pts DESC LIMIT 30;"},
    {"question": "How has teveslage been trending this season?", "sql": "SELECT e.event_name, e.date, ROUND(SUM(dt.total_pts)::numeric,1) AS total_pts FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE a.username='teveslage' AND e.calendar_year=2026 GROUP BY e.event_name,e.date ORDER BY e.date DESC LIMIT 100;"},
    {"question": "Which golfers were most commonly picked across all managers this year?", "sql": "SELECT SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer, COUNT(*) AS times_selected FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN player p ON p.dg_id=dt.dg_id JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY p.player_name ORDER BY times_selected DESC LIMIT 20;"},
    {"question": "Who has earned the most money this season?", "sql": "WITH scores AS (SELECT a.username, e.id_event, e.dfs_payout, SUM(dt.total_pts) AS pts FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 AND e.dfs_payout IS NOT NULL GROUP BY a.username,e.id_event,e.dfs_payout), ranked AS (SELECT *, RANK() OVER (PARTITION BY id_event ORDER BY pts DESC) AS rnk FROM scores) SELECT username, ROUND(SUM(dfs_payout)::numeric,2) AS total_earned, COUNT(*) AS wins FROM ranked WHERE rnk=1 GROUP BY username ORDER BY total_earned DESC;"},
    {"question": "What is the next event coming up?", "sql": "SELECT event_name, ROUND(AVG(EXTRACT(DOY FROM date))) AS typical_day_of_year, MIN(calendar_year) AS first_seen FROM event GROUP BY event_name ORDER BY typical_day_of_year ASC LIMIT 10;"},
    {"question": "What was cjt3's performance at the Masters in previous years?", "sql": "SELECT a.username, e.event_name, e.calendar_year, ROUND(SUM(dt.total_pts)::numeric,1) AS dfs_total FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.event_name LIKE '%Masters%' AND a.username='cjt3' GROUP BY e.calendar_year,e.event_name,a.username ORDER BY e.calendar_year DESC;"},
    {"question": "Who has missed the most events or forgotten to set their lineup?", "sql": "WITH cte1 AS (SELECT a.username, e.event_name, e.calendar_year, SUM(dt.total_pts) AS dfs_count FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY e.calendar_year,e.event_name,a.username), cte2 AS (SELECT username, COUNT(DISTINCT event_name) AS unique_dfs FROM cte1 GROUP BY username), cte3 AS (SELECT COUNT(DISTINCT event_name) AS unique_events FROM cte1) SELECT username, unique_events-unique_dfs AS missed_events FROM cte2 CROSS JOIN cte3 ORDER BY missed_events DESC;"},
    {"question": "Show me total counts of missed events where managers forgot to set lineups in 2026", "sql": "WITH cte1 AS (SELECT a.username, e.event_name, e.calendar_year, SUM(dt.total_pts) AS dfs_count FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY e.calendar_year,e.event_name,a.username), cte2 AS (SELECT username, COUNT(DISTINCT event_name) AS unique_dfs FROM cte1 GROUP BY username), cte3 AS (SELECT COUNT(DISTINCT event_name) AS unique_events FROM cte1) SELECT username, unique_events-unique_dfs AS missed_events FROM cte2 CROSS JOIN cte3 ORDER BY missed_events DESC;"},
    {"question": "Which managers have not submitted lineups for all events?", "sql": "WITH cte1 AS (SELECT a.username, e.event_name, e.calendar_year, SUM(dt.total_pts) AS dfs_count FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY e.calendar_year,e.event_name,a.username), cte2 AS (SELECT username, COUNT(DISTINCT event_name) AS unique_dfs FROM cte1 GROUP BY username), cte3 AS (SELECT COUNT(DISTINCT event_name) AS unique_events FROM cte1) SELECT username, unique_events-unique_dfs AS missed_events FROM cte2 CROSS JOIN cte3 ORDER BY missed_events DESC;"},
    {"question": "Who forgot to set their DFS lineup this season?", "sql": "WITH cte1 AS (SELECT a.username, e.event_name, e.calendar_year, SUM(dt.total_pts) AS dfs_count FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY e.calendar_year,e.event_name,a.username), cte2 AS (SELECT username, COUNT(DISTINCT event_name) AS unique_dfs FROM cte1 GROUP BY username), cte3 AS (SELECT COUNT(DISTINCT event_name) AS unique_events FROM cte1) SELECT username, unique_events-unique_dfs AS missed_events FROM cte2 CROSS JOIN cte3 ORDER BY missed_events DESC;"}
]

_vn = None

def get_secret(secret_name):
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

def get_vanna(api_key, db_config=None):
    global _vn
    if _vn is not None:
        return _vn

    class PGAVanna(ChromaDB_VectorStore, Anthropic_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            Anthropic_Chat.__init__(self, config=config)

    import chromadb
    chroma_client = chromadb.PersistentClient(path="/tmp/chromadb")
    _vn = PGAVanna(config={
        "api_key": api_key,
        "model": "claude-sonnet-4-20250514",
        "client": chroma_client
    })
    for ex in TRAINING_EXAMPLES:
        _vn.train(question=ex["question"], sql=ex["sql"])

    # Pull approved examples from RDS dynamically
    if db_config:
        try:
            import psycopg2
            pg = psycopg2.connect(**db_config)
            pg_cur = pg.cursor()
            pg_cur.execute("SELECT question, sql_query FROM training_log WHERE approved = true ORDER BY created_at ASC")
            approved = pg_cur.fetchall()
            pg_cur.close()
            pg.close()
            for question, sql in approved:
                _vn.train(question=question, sql=sql)
            print(f"[pga-vanna] Loaded {len(approved)} approved examples from RDS")
        except Exception as e:
            print(f"[pga-vanna] Could not load approved examples: {e}")

    _vn.train(ddl="""
        CREATE TABLE account (id_account UUID PRIMARY KEY, first_name TEXT, last_name TEXT, username TEXT, active BOOLEAN);
        CREATE TABLE event (id_event UUID PRIMARY KEY, calendar_year INT, event_id INT, date DATE, event_name TEXT, dfs_payout NUMERIC);
        CREATE TABLE player (dg_id INT PRIMARY KEY, player_name TEXT, country TEXT, country_code TEXT, amateur INT);
        CREATE TABLE dfs_total (id_dfs UUID PRIMARY KEY, id_event UUID REFERENCES event, dg_id INT REFERENCES player, fin_text TEXT, total_pts FLOAT, salary INT, hole_score_pts FLOAT, finish_pts INT, five_birdie_pts INT, bogey_free_pts INT, bounce_back_pts FLOAT, streak_pts FLOAT);
        CREATE TABLE dfs_board (id_board UUID PRIMARY KEY, id_account UUID REFERENCES account, id_dfs_1 UUID REFERENCES dfs_total, id_dfs_2 UUID REFERENCES dfs_total, id_dfs_3 UUID REFERENCES dfs_total, id_dfs_4 UUID REFERENCES dfs_total, id_dfs_5 UUID REFERENCES dfs_total, id_dfs_6 UUID REFERENCES dfs_total);
        CREATE TABLE course (id_course UUID PRIMARY KEY, course_name TEXT, course_num INT, course_par INT);
        CREATE TABLE round (id_round UUID PRIMARY KEY, id_event UUID REFERENCES event, id_course UUID REFERENCES course, dg_id INT REFERENCES player, round FLOAT, score INT, pars INT, birdies INT, bogies INT, doubles_or_worse INT, eagles_or_better INT, sg_total FLOAT, sg_t2g FLOAT, sg_putt FLOAT, sg_ott FLOAT, sg_arg FLOAT, sg_app FLOAT, scrambling FLOAT, driving_dist FLOAT, driving_acc FLOAT, gir FLOAT);
        CREATE TABLE training_log (id UUID PRIMARY KEY, question TEXT, sql_query TEXT, approved BOOLEAN, created_at TIMESTAMP);
    """)

    _vn.train(documentation="""
        dfs_board has 6 pick columns (id_dfs_1 through id_dfs_6). To unnest picks ALWAYS use:
        JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true
        JOIN dfs_total dt ON dt.id_dfs = picks.id_dfs
        NEVER use IN (id_dfs_1, id_dfs_2, id_dfs_3, id_dfs_4, id_dfs_5, id_dfs_6).
        player_name is stored as LastName, FirstName. Display using SPLIT_PART(player_name, ', ', 2) || ' ' || SPLIT_PART(player_name, ', ', 1).
        fin_text CUT and WD mean the player did not complete the tournament.
        Active managers only: WHERE account.active = true.
    """)

    return _vn

def respond(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body)
    }

def lambda_handler(event, context):
    try:
        if event.get("httpMethod") == "OPTIONS":
            return respond(200, {})

        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "").strip()

        if not question:
            return respond(400, {"error": "question required"})

        print(f"[pga-vanna] Question received: {question}")

        secrets = get_secret(os.environ.get("SECRET_NAME", "pga-db-sync-secret"))
        api_key = secrets.get("anthropic_key")
        if not api_key:
            return respond(500, {"error": "anthropic_key not found"})

        db_config = {
            "host": secrets.get("host"),
            "port": int(secrets.get("port", 5432)),
            "dbname": secrets.get("dbname", "postgres"),
            "user": secrets.get("username"),
            "password": secrets.get("password")
        }

        vn = get_vanna(api_key, db_config=db_config)
        print(f"[pga-vanna] Vanna initialized, generating SQL...")
        sql = vn.generate_sql(question)
        print(f"[pga-vanna] SQL generated: {sql[:200]}")

        return respond(200, {"sql": sql, "question": question})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return respond(500, {"error": str(e)})
