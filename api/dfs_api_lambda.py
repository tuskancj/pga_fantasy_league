import json
import os
import boto3
import psycopg2
import psycopg2.extras
import urllib.request
from decimal import Decimal
from datetime import date, datetime


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def get_connection():
    secret_name = os.environ.get("SECRET_NAME")
    if secret_name:
        s = get_secret(secret_name)
        cfg = dict(dbname=s.get("dbname","postgres"), user=s["username"],
                   password=s["password"], host=s["host"], port=s.get("port","5432"))
    else:
        cfg = dict(dbname=os.environ.get("DB_NAME","postgres"),
                   user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
                   host=os.environ["DB_HOST"], port=os.environ.get("DB_PORT","5432"))
    return psycopg2.connect(**cfg)


def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Content-Type": "application/json"
    }


def respond(status, body):
    return {"statusCode": status, "headers": cors_headers(), "body": json.dumps(body, cls=CustomEncoder)}


def lambda_handler(event, context):
    # Support both REST API (httpMethod/path) and HTTP API (requestContext.http)
    rc = event.get("requestContext", {})
    http = rc.get("http", {})
    method = http.get("method") or event.get("httpMethod", "GET")
    path   = http.get("path") or event.get("path", "/")
    params = event.get("queryStringParameters") or {}

    if method == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers(), "body": ""}

    try:
        conn = get_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # GET /managers
        if path == "/managers" and method == "GET":
            cur.execute("""
                SELECT id_account::text, username
                FROM account
                WHERE active = true
                ORDER BY username
            """)
            return respond(200, cur.fetchall())

        # GET /players
        if path == "/players" and method == "GET":
            cur.execute("SELECT dg_id, player_name FROM player ORDER BY player_name")
            return respond(200, cur.fetchall())

        # GET /events?year=YYYY
        if path == "/events" and method == "GET":
            year = params.get("year", datetime.now().year)
            cur.execute("""
                SELECT
                    e.id_event::text,
                    e.event_id,
                    e.event_name,
                    e.date::text,
                    e.calendar_year,
                    e.dfs_payout,
                    EXISTS (
                        SELECT 1 FROM dfs_board db
                        INNER JOIN dfs_total dt ON dt.id_dfs = db.id_dfs_1
                        WHERE dt.id_event = e.id_event
                        LIMIT 1
                    ) AS has_board
                FROM event e
                WHERE e.calendar_year = %(year)s
                ORDER BY e.date DESC
            """, {"year": year})
            return respond(200, cur.fetchall())

        # GET /dfs-players?id_event=UUID
        if path == "/dfs-players" and method == "GET":
            id_event = params.get("id_event")
            if not id_event:
                return respond(400, {"error": "id_event required"})
            cur.execute("""
                SELECT
                    dt.id_dfs::text,
                    p.player_name,
                    dt.total_pts,
                    dt.salary,
                    dt.fin_text
                FROM dfs_total dt
                INNER JOIN player p ON p.dg_id = dt.dg_id
                WHERE dt.id_event = %(id_event)s
                ORDER BY p.player_name
            """, {"id_event": id_event})
            return respond(200, cur.fetchall())

        # GET /board?id_event=UUID
        if path == "/board" and method == "GET":
            id_event = params.get("id_event")
            if not id_event:
                return respond(400, {"error": "id_event required"})
            cur.execute("""
                SELECT
                    a.username,
                    p1.player_name AS p1,
                    p2.player_name AS p2,
                    p3.player_name AS p3,
                    p4.player_name AS p4,
                    p5.player_name AS p5,
                    p6.player_name AS p6
                FROM dfs_board db
                INNER JOIN account a ON a.id_account = db.id_account
                LEFT JOIN dfs_total dt1 ON dt1.id_dfs = db.id_dfs_1
                LEFT JOIN player p1 ON p1.dg_id = dt1.dg_id
                LEFT JOIN dfs_total dt2 ON dt2.id_dfs = db.id_dfs_2
                LEFT JOIN player p2 ON p2.dg_id = dt2.dg_id
                LEFT JOIN dfs_total dt3 ON dt3.id_dfs = db.id_dfs_3
                LEFT JOIN player p3 ON p3.dg_id = dt3.dg_id
                LEFT JOIN dfs_total dt4 ON dt4.id_dfs = db.id_dfs_4
                LEFT JOIN player p4 ON p4.dg_id = dt4.dg_id
                LEFT JOIN dfs_total dt5 ON dt5.id_dfs = db.id_dfs_5
                LEFT JOIN player p5 ON p5.dg_id = dt5.dg_id
                LEFT JOIN dfs_total dt6 ON dt6.id_dfs = db.id_dfs_6
                LEFT JOIN player p6 ON p6.dg_id = dt6.dg_id
                INNER JOIN dfs_total dt_any ON dt_any.id_dfs = db.id_dfs_1
                WHERE dt_any.id_event = %(id_event)s
            """, {"id_event": id_event})
            return respond(200, cur.fetchall())

        # POST /submit-board
        if path == "/submit-board" and method == "POST":
            body = json.loads(event.get("body", "{}"))
            id_event = body.get("id_event")
            payout   = body.get("payout")
            board    = body.get("board", [])

            if not id_event or not board:
                return respond(400, {"error": "id_event and board required"})

            # Delete existing board entries for this event if re-submitting
            cur.execute("""
                DELETE FROM dfs_board
                WHERE id_account IN (
                    SELECT id_account FROM account WHERE active = true
                )
                AND id_dfs_1 IN (
                    SELECT id_dfs FROM dfs_total WHERE id_event = %(id_event)s
                )
            """, {"id_event": id_event})

            # Insert new board rows
            for row in board:
                cur.execute("""
                    INSERT INTO dfs_board
                        (id_account, id_dfs_1, id_dfs_2, id_dfs_3, id_dfs_4, id_dfs_5, id_dfs_6)
                    VALUES
                        (%(id_account)s, %(id_dfs_1)s, %(id_dfs_2)s, %(id_dfs_3)s,
                         %(id_dfs_4)s, %(id_dfs_5)s, %(id_dfs_6)s)
                """, row)

            # Update payout if provided and not already set
            if payout is not None:
                cur.execute("""
                    UPDATE event
                    SET dfs_payout = %(payout)s
                    WHERE id_event = %(id_event)s
                    AND dfs_payout IS NULL
                """, {"payout": payout, "id_event": id_event})

            conn.commit()
            return respond(200, {"success": True, "rows_inserted": len(board)})

        # POST /insert-wd
        if path == "/insert-wd" and method == "POST":
            body     = json.loads(event.get("body", "{}"))
            dg_id    = body.get("dg_id")
            id_event = body.get("id_event")
            salary   = body.get("salary", 0)

            if not all([dg_id, id_event]):
                return respond(400, {"error": "dg_id and id_event are required"})

            cur.execute("""
                INSERT INTO dfs_total (
                    id_event, dg_id,
                    bogey_free_pts, bounce_back_pts, fin_text, finish_pts,
                    five_birdie_pts, hole_score_pts, salary, streak_pts, total_pts
                )
                VALUES (%(id_event)s, %(dg_id)s, 0, 0, 'WD', 0, 0, 0, %(salary)s, 0, 0)
            """, {"dg_id": dg_id, "id_event": id_event, "salary": salary})

            conn.commit()
            return respond(200, {"success": True})

        # GET /dashboard?year=YYYY
        if path == "/dashboard" and method == "GET":
            year = int(params.get("year", datetime.now().year))

            # 1. Standings — total points per manager for the season
            cur.execute("""
                SELECT
                    a.username,
                    ROUND(SUM(dt.total_pts)::numeric, 1) AS total_pts,
                    COUNT(DISTINCT e.id_event) AS events_played
                FROM dfs_board db
                INNER JOIN account a ON a.id_account = db.id_account
                INNER JOIN dfs_total dt ON dt.id_dfs IN (
                    db.id_dfs_1, db.id_dfs_2, db.id_dfs_3,
                    db.id_dfs_4, db.id_dfs_5, db.id_dfs_6
                )
                INNER JOIN event e ON e.id_event = dt.id_event
                WHERE e.calendar_year = %(year)s
                GROUP BY a.username
                ORDER BY total_pts DESC
            """, {"year": year})
            standings = cur.fetchall()

            # 2. Pick frequency — how many times each golfer was selected
            cur.execute("""
                SELECT
                    p.player_name,
                    COUNT(*) AS times_selected
                FROM dfs_board db
                INNER JOIN dfs_total dt ON dt.id_dfs IN (
                    db.id_dfs_1, db.id_dfs_2, db.id_dfs_3,
                    db.id_dfs_4, db.id_dfs_5, db.id_dfs_6
                )
                INNER JOIN player p ON p.dg_id = dt.dg_id
                INNER JOIN event e ON e.id_event = dt.id_event
                WHERE e.calendar_year = %(year)s
                GROUP BY p.player_name
                ORDER BY times_selected DESC
                LIMIT 30
            """, {"year": year})
            pick_frequency = cur.fetchall()

            # 3. Salary vs points — aggregate per golfer across all events
            cur.execute("""
                SELECT
                    p.player_name,
                    ROUND(AVG(dt.salary)::numeric, 0) AS avg_salary,
                    ROUND(AVG(dt.total_pts)::numeric, 2) AS avg_pts,
                    COUNT(*) AS events_played
                FROM dfs_total dt
                INNER JOIN player p ON p.dg_id = dt.dg_id
                INNER JOIN event e ON e.id_event = dt.id_event
                WHERE e.calendar_year = %(year)s
                AND dt.salary > 0
                AND dt.total_pts > 0
                GROUP BY p.player_name
                HAVING COUNT(*) >= 1
                ORDER BY avg_pts DESC
            """, {"year": year})
            salary_vs_pts = cur.fetchall()

            # 4. Trending managers — last 3 events, points per event
            cur.execute("""
                SELECT
                    a.username,
                    e.event_name,
                    e.date,
                    ROUND(SUM(dt.total_pts)::numeric, 1) AS total_pts
                FROM dfs_board db
                INNER JOIN account a ON a.id_account = db.id_account
                INNER JOIN dfs_total dt ON dt.id_dfs IN (
                    db.id_dfs_1, db.id_dfs_2, db.id_dfs_3,
                    db.id_dfs_4, db.id_dfs_5, db.id_dfs_6
                )
                INNER JOIN event e ON e.id_event = dt.id_event
                WHERE e.calendar_year = %(year)s
                AND e.id_event IN (
                    SELECT id_event FROM event
                    WHERE calendar_year = %(year)s
                    AND id_event IN (
                        SELECT DISTINCT dt2.id_event FROM dfs_total dt2
                        INNER JOIN dfs_board db2 ON dt2.id_dfs IN (
                            db2.id_dfs_1, db2.id_dfs_2, db2.id_dfs_3,
                            db2.id_dfs_4, db2.id_dfs_5, db2.id_dfs_6
                        )
                    )
                    ORDER BY date DESC
                    LIMIT 3
                )
                GROUP BY a.username, e.event_name, e.date
                ORDER BY e.date ASC, total_pts DESC
            """, {"year": year})
            trending_managers = cur.fetchall()

            # 5. Trending golfers — last 3 events FanDuel scoring
            cur.execute("""
                SELECT
                    p.player_name,
                    e.event_name,
                    e.date,
                    ROUND(dt.total_pts::numeric, 1) AS total_pts,
                    dt.fin_text
                FROM dfs_total dt
                INNER JOIN player p ON p.dg_id = dt.dg_id
                INNER JOIN event e ON e.id_event = dt.id_event
                WHERE e.calendar_year = %(year)s
                AND e.id_event IN (
                    SELECT id_event FROM event
                    WHERE calendar_year = %(year)s
                    AND id_event IN (
                        SELECT DISTINCT dt2.id_event FROM dfs_total dt2
                        INNER JOIN dfs_board db2 ON dt2.id_dfs IN (
                            db2.id_dfs_1, db2.id_dfs_2, db2.id_dfs_3,
                            db2.id_dfs_4, db2.id_dfs_5, db2.id_dfs_6
                        )
                    )
                    ORDER BY date DESC
                    LIMIT 3
                )
                ORDER BY e.date ASC, dt.total_pts DESC
            """, {"year": year})
            trending_golfers = cur.fetchall()

            # 6. Per-event scores for each manager (for rank trend tab)
            cur.execute("""
                SELECT
                    a.username,
                    e.event_name,
                    e.date,
                    ROUND(SUM(dt.total_pts)::numeric, 1) AS total_pts
                FROM dfs_board db
                INNER JOIN account a ON a.id_account = db.id_account
                INNER JOIN dfs_total dt ON dt.id_dfs IN (
                    db.id_dfs_1, db.id_dfs_2, db.id_dfs_3,
                    db.id_dfs_4, db.id_dfs_5, db.id_dfs_6
                )
                INNER JOIN event e ON e.id_event = dt.id_event
                WHERE e.calendar_year = %(year)s
                GROUP BY a.username, e.event_name, e.date
                ORDER BY e.date ASC
            """, {"year": year})
            manager_event_scores = cur.fetchall()

            # 7. Winner earnings — top scorer per event gets the payout
            cur.execute("""
                WITH event_scores AS (
                    SELECT
                        a.username,
                        e.id_event,
                        e.event_name,
                        e.date,
                        e.dfs_payout,
                        ROUND(SUM(dt.total_pts)::numeric, 1) AS total_pts,
                        RANK() OVER (PARTITION BY e.id_event ORDER BY SUM(dt.total_pts) DESC) AS rank
                    FROM dfs_board db
                    INNER JOIN account a ON a.id_account = db.id_account
                    INNER JOIN dfs_total dt ON dt.id_dfs IN (
                        db.id_dfs_1, db.id_dfs_2, db.id_dfs_3,
                        db.id_dfs_4, db.id_dfs_5, db.id_dfs_6
                    )
                    INNER JOIN event e ON e.id_event = dt.id_event
                    WHERE e.calendar_year = %(year)s
                    AND e.dfs_payout IS NOT NULL
                    GROUP BY a.username, e.id_event, e.event_name, e.date, e.dfs_payout
                )
                SELECT
                    username,
                    ROUND(SUM(dfs_payout)::numeric, 2) AS total_earnings,
                    COUNT(*) AS events_won
                FROM event_scores
                WHERE rank = 1
                GROUP BY username
                ORDER BY total_earnings DESC
            """, {"year": year})
            winner_earnings = cur.fetchall()

            # 7b. No comment — just needed for stat card
            cur.execute("""
                WITH event_scores AS (
                    SELECT
                        a.username,
                        e.id_event,
                        e.dfs_payout,
                        ROUND(SUM(dt.total_pts)::numeric, 1) AS total_pts,
                        RANK() OVER (PARTITION BY e.id_event ORDER BY SUM(dt.total_pts) DESC) AS rank
                    FROM dfs_board db
                    INNER JOIN account a ON a.id_account = db.id_account
                    INNER JOIN dfs_total dt ON dt.id_dfs IN (
                        db.id_dfs_1, db.id_dfs_2, db.id_dfs_3,
                        db.id_dfs_4, db.id_dfs_5, db.id_dfs_6
                    )
                    INNER JOIN event e ON e.id_event = dt.id_event
                    WHERE e.calendar_year = %(year)s
                    AND e.dfs_payout IS NOT NULL
                    GROUP BY a.username, e.id_event, e.dfs_payout
                )
                SELECT username, SUM(dfs_payout) AS total_earnings
                FROM event_scores WHERE rank = 1
                GROUP BY username ORDER BY total_earnings DESC LIMIT 1
            """, {"year": year})
            top_earner = cur.fetchone()

            # 7. Multi-year golfer history for chat — no longer sent in payload
            # Chat now uses SQL generation to query directly

            return respond(200, {
                "year": year,
                "standings": standings,
                "pick_frequency": pick_frequency,
                "salary_vs_pts": salary_vs_pts,
                "trending_managers": trending_managers,
                "trending_golfers": trending_golfers,
                "manager_event_scores": manager_event_scores,
                "winner_earnings": winner_earnings,
                "top_earner": top_earner
            })

        # POST /chat
        if path == "/chat" and method == "POST":
            body     = json.loads(event.get("body", "{}"))
            messages = body.get("messages", [])
            context  = body.get("context", {})  # year, standings summary etc from frontend

            if not messages:
                return respond(400, {"error": "messages required"})

            secret_name = os.environ.get("SECRET_NAME")
            if not secret_name:
                return respond(500, {"error": "SECRET_NAME not configured"})

            secrets = get_secret(secret_name)
            anthropic_key = secrets.get("anthropic_key")
            if not anthropic_key:
                return respond(500, {"error": "anthropic_key not found in secret"})

            def call_claude(system, msgs, max_tokens=1000):
                payload = json.dumps({
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": msgs
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req) as r:
                    return json.loads(r.read().decode("utf-8"))

            schema_prompt = """You are a data analyst assistant for a PGA FanDuel fantasy golf league.

DATABASE SCHEMA (PostgreSQL):
account(id_account UUID PK, first_name, last_name, username, active BOOLEAN)
event(id_event UUID PK, calendar_year INT, event_id INT, date DATE, event_name, dfs_payout NUMERIC)
player(dg_id INT PK, player_name, country, country_code, amateur INT)
  - player_name format: "LastName, FirstName"
dfs_total(id_dfs UUID PK, id_event UUID FK, dg_id INT FK, fin_text, total_pts FLOAT,
  salary INT, hole_score_pts FLOAT, finish_pts INT, five_birdie_pts INT,
  bogey_free_pts INT, bounce_back_pts FLOAT, streak_pts FLOAT)
  - fin_text: finish position e.g. "T5", "1", "CUT", "WD"
dfs_board(id_board UUID PK, id_account UUID FK, id_dfs_1..id_dfs_6 UUID FK -> dfs_total.id_dfs)
  - each row = one manager's 6 golfer picks for one event
  - ALWAYS unnest picks with: JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true
  - Then JOIN dfs_total dt ON dt.id_dfs = picks.id_dfs
  - NEVER join dfs_total 6 times or use IN (id_dfs_1,...,id_dfs_6) for pick unnesting
course(id_course UUID PK, course_name, course_num INT, course_par INT)
round(id_round UUID PK, id_event UUID FK, id_course UUID FK, dg_id INT FK,
  round FLOAT, score INT, pars INT, birdies INT, bogies INT, doubles_or_worse INT,
  eagles_or_better INT, sg_total FLOAT, sg_t2g FLOAT, sg_putt FLOAT,
  sg_ott FLOAT, sg_arg FLOAT, sg_app FLOAT, scrambling FLOAT,
  driving_dist FLOAT, driving_acc FLOAT, gir FLOAT)

Current context: """ + json.dumps(context)

            sql_system = schema_prompt + """

TASK: Translate the user's question into a single PostgreSQL SELECT query.

RULES:
- Output ONLY the SQL query. No markdown, no backticks, no explanation.
- ALWAYS query the database first. Every question about managers, golfers, picks, events, scores, or performance CAN be answered from this schema.
- Only respond with CANNOT_QUERY if the question is completely unrelated to golf/DFS (e.g. "what's the weather")
- LIMIT results to 100 rows maximum
- Round numeric results to 1-2 decimal places
- For pick-related questions about a specific manager: filter by account.username
- player_name is "LastName, FirstName" — use SPLIT_PART or CONCAT to reformat for display
- fin_text 'WD' or 'CUT' = did not complete tournament
- active managers only: JOIN account WHERE account.active = true"""

            # Step 1: Generate SQL
            sql_response = call_claude(sql_system, messages, max_tokens=700)
            sql_query = sql_response["content"][0]["text"].strip()

            if sql_query == "CANNOT_QUERY" or sql_query.upper().startswith("CANNOT"):
                answer_system = schema_prompt + """
Answer the user's question using your general PGA/golf/FanDuel knowledge.
Be concise. NEVER mention SQL, databases, or ask the user to do anything technical.
If they want a chart respond with:
CHART_JSON:{"type":"bar|line|scatter","title":"...","labels":[...],"datasets":[{"label":"...","data":[...],"color":"#hex"}]}"""
                answer_response = call_claude(answer_system, messages)
                return respond(200, {
                    "content": answer_response["content"],
                    "sql": None,
                    "results": None
                })

            # Step 2: Execute SQL — auto-retry once with Claude fixing the query
            results = []
            for attempt in range(2):
                try:
                    sql_cur = conn.cursor()
                    sql_cur.execute(sql_query)
                    col_names = [desc[0] for desc in sql_cur.description]
                    rows = sql_cur.fetchall()
                    results = [dict(zip(col_names, row)) for row in rows]
                    sql_cur.close()
                    break
                except Exception as sql_err:
                    try: sql_cur.close()
                    except: pass
                    conn.rollback()
                    if attempt == 0:
                        fix_system = schema_prompt + f"""
The following SQL failed: {sql_query}
Error: {str(sql_err)}

Write a corrected SQL query. Output ONLY the SQL — no markdown, no explanation.
Key reminder: for dfs_board picks ALWAYS use LATERAL VALUES to unnest, never join each id_dfs_N separately."""
                        fix_response = call_claude(fix_system, messages, max_tokens=700)
                        sql_query = fix_response["content"][0]["text"].strip()
                    else:
                        sql_query = f"SQL_ERROR: {str(sql_err)}"
                        results = []

            # Step 3: Interpret and respond naturally — user never sees SQL
            results_str = json.dumps(results[:75], cls=CustomEncoder)
            interpret_system = schema_prompt + f"""

You just looked up data from the database to answer the user's question.
Here are the results ({len(results)} rows total, showing up to 75):
{results_str}

Respond naturally and conversationally — as if you just looked this up yourself.
NEVER mention SQL, databases, queries, or ask the user to do anything technical.
NEVER say you "ran a query" or "checked the database".
Just answer the question directly with the data.

If results are empty, say you couldn't find that information and suggest rephrasing.
If they want a chart, respond with (text explanation first, then on its own line):
CHART_JSON:{{"type":"bar|line|scatter","title":"...","labels":[...],"datasets":[{{"label":"...","data":[...],"color":"#hex"}}]}}"""

            answer_response = call_claude(interpret_system, messages)

            return respond(200, {
                "content": answer_response["content"],
                "sql": sql_query,
                "results_count": len(results)
            })

        return respond(404, {"error": f"Unknown path: {path}"})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return respond(500, {"error": str(e)})

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass
