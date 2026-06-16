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

            # 1. Standings - total points per manager for the season
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

            # 2. Pick frequency - how many times each golfer was selected
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

            # 3. Salary vs points - aggregate per golfer across all events
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

            # 4. Trending managers - last 3 events, points per event
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

            # 5. Trending golfers - last 3 events FanDuel scoring
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

            # 7. Winner earnings - top scorer per event gets the payout
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

            # 7b. No comment - just needed for stat card
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

            # 7. Multi-year golfer history for chat - no longer sent in payload
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

        # POST /chat - Mistral Large via Bedrock
        if path == "/chat" and method == "POST":
            body = json.loads(event.get("body", "{}"))
            messages = body.get("messages", [])
            model_id = body.get("model_id", "mistral.mistral-large-2402-v1:0")

            if not messages:
                return respond(400, {"error": "messages required"})

            question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

            prior = [m for m in messages[:-1] if m["role"] == "user"]
            new_subject_signals = ['performance', 'history', 'show', 'who', 'what', 'how', 'give', 'list', 'compare', 'pick', 'win', 'miss', 'score', 'salary']
            is_followup = (
                len(question.split()) <= 3 and
                not any(sig in question.lower() for sig in new_subject_signals)
            )
            contextual_question = f"{prior[-1]['content']} - specifically {question}" if (is_followup and prior) else question

            history_str = ""
            for m in messages[:-1][-6:]:
                role = "User" if m["role"] == "user" else "Assistant"
                history_str += f"{role}: {m['content'][:200]}\n"

            print(f"[chat] Question: {contextual_question}")

            secrets = get_secret(os.environ.get("SECRET_NAME"))
            current_year = __import__("datetime").datetime.now().year

            examples = """Q: What are cjt3's most picked golfers?
SELECT SPLIT_PART(p.player_name,', ',2)||\' \'||SPLIT_PART(p.player_name,', ',1) AS golfer, COUNT(*) AS times_picked FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN player p ON p.dg_id=dt.dg_id WHERE a.username='cjt3' GROUP BY p.player_name ORDER BY times_picked DESC LIMIT 20;

Q: Who won the most events in 2026?
WITH scores AS (SELECT a.username, e.id_event, SUM(dt.total_pts) AS pts FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY a.username,e.id_event), ranked AS (SELECT *, RANK() OVER (PARTITION BY id_event ORDER BY pts DESC) AS rnk FROM scores) SELECT username, COUNT(*) AS wins FROM ranked WHERE rnk=1 GROUP BY username ORDER BY wins DESC;

Q: What are the current season standings?
SELECT a.username, ROUND(SUM(dt.total_pts)::numeric,1) AS total_pts, COUNT(DISTINCT e.id_event) AS events FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 AND a.active=true GROUP BY a.username ORDER BY total_pts DESC;

Q: Who has missed the most events?
WITH cte1 AS (SELECT a.username, e.event_name, e.calendar_year, SUM(dt.total_pts) AS dfs_count FROM dfs_board db JOIN account a ON db.id_account=a.id_account JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true JOIN dfs_total dt ON dt.id_dfs=picks.id_dfs JOIN event e ON e.id_event=dt.id_event WHERE e.calendar_year=2026 GROUP BY e.calendar_year,e.event_name,a.username), cte2 AS (SELECT username, COUNT(DISTINCT event_name) AS unique_dfs FROM cte1 GROUP BY username), cte3 AS (SELECT COUNT(DISTINCT event_name) AS unique_events FROM cte1) SELECT username, unique_events-unique_dfs AS missed_events FROM cte2 CROSS JOIN cte3 ORDER BY missed_events DESC;"""

            try:
                tl_cur = conn.cursor()
                tl_cur.execute("SELECT question, sql_query FROM training_log WHERE approved = true ORDER BY created_at ASC LIMIT 30")
                for q, s in tl_cur.fetchall():
                    examples += f"\n\nQ: {q}\n{s}"
                tl_cur.close()
            except Exception:
                conn.rollback()

            sql_prompt = f"""You are a PostgreSQL SQL expert for a PGA FanDuel fantasy golf league.

SCHEMA:
account(id_account UUID PK, username, active BOOLEAN)
event(id_event UUID PK, calendar_year INT, event_id INT, date DATE, event_name TEXT, dfs_payout NUMERIC)
player(dg_id INT PK, player_name TEXT) -- player_name: LastName, FirstName
dfs_total(id_dfs UUID PK, id_event UUID FK, dg_id INT FK, fin_text TEXT, total_pts FLOAT, salary INT, hole_score_pts FLOAT, finish_pts INT, five_birdie_pts INT, bogey_free_pts INT, bounce_back_pts FLOAT, streak_pts FLOAT)
dfs_board(id_board UUID PK, id_account UUID FK, id_dfs_1..id_dfs_6 UUID FK -> dfs_total.id_dfs)
round(id_round UUID PK, id_event UUID FK, id_course UUID FK, dg_id INT FK, round FLOAT, score INT, sg_total FLOAT, sg_putt FLOAT, sg_ott FLOAT, sg_arg FLOAT, sg_app FLOAT)
model_run(id_run UUID PK, id_event UUID FK, model_version TEXT, run_timestamp TIMESTAMPTZ, config JSONB)
prediction_golfer(id_run UUID FK, dg_id INT FK, salary INT, sim_mean NUMERIC, sim_sd NUMERIC, sim_p_low NUMERIC, sim_p_high NUMERIC, made_cut_pct NUMERIC, dg_proj NUMERIC)
prediction_lineup(id_lineup UUID PK, id_run UUID FK, tier TEXT, predicted_total NUMERIC, dg_proj_total NUMERIC, salary_total INT)
prediction_lineup_pick(id_lineup UUID FK, dg_id INT FK)

CRITICAL: For dfs_board picks ALWAYS use LATERAL VALUES:
JOIN LATERAL (VALUES (db.id_dfs_1),(db.id_dfs_2),(db.id_dfs_3),(db.id_dfs_4),(db.id_dfs_5),(db.id_dfs_6)) AS picks(id_dfs) ON true
JOIN dfs_total dt ON dt.id_dfs = picks.id_dfs

RULES: Output ONLY raw SQL. LIMIT 100 rows. Display names: SPLIT_PART(player_name,', ',2)||\' \'||SPLIT_PART(player_name,', ',1). Active managers: WHERE account.active = true.

EXAMPLES:
{{examples}}

{{"Prior conversation:" + chr(10) + history_str if history_str else ""}}
Write SQL for: {{contextual_question}}"""

            sql_prompt = sql_prompt.replace("{examples}", examples).replace("{contextual_question}", contextual_question)

            bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
            sql_response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({"prompt": f"<s>[INST]{sql_prompt}[/INST]", "max_tokens": 1000, "temperature": 0.1}),
                contentType="application/json", accept="application/json"
            )
            sql_query = json.loads(sql_response["body"].read()).get("outputs", [{}])[0].get("text", "").strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            print(f"[chat] SQL: {sql_query[:200]}")

            results, col_names, sql_failed = [], [], False
            try:
                sql_cur = conn.cursor()
                sql_cur.execute(sql_query)
                col_names = [desc[0] for desc in sql_cur.description]
                results = [dict(zip(col_names, row)) for row in sql_cur.fetchall()]
                sql_cur.close()
            except Exception as sql_err:
                try: sql_cur.close()
                except: pass
                conn.rollback()
                sql_failed = True
                sql_query = f"SQL_ERROR: {str(sql_err)}"

            try:
                log_cur = conn.cursor()
                log_cur.execute("INSERT INTO training_log (question, sql_query, approved, source) VALUES (%s, %s, false, %s)", (question, sql_query, "mistral"))
                conn.commit()
                log_cur.close()
            except Exception:
                conn.rollback()

            if sql_failed:
                return respond(200, {"content": [{"type": "text", "text": "I had trouble with that question. Could you try rephrasing it?"}], "sql": sql_query, "results_count": 0})

            results_str = json.dumps(results[:75], cls=CustomEncoder)
            col_summary = ", ".join(col_names) if col_names else ""

            interpret_prompt = f"""You are a data analyst for a PGA FanDuel fantasy golf league. Current year: {current_year}.

{{"Prior conversation:" + chr(10) + history_str if history_str else ""}}
The user asked: "{contextual_question}"

Result columns: {col_summary}
Results ({len(results)} rows):
{results_str}

AMBIGUITY: If multiple distinct players match a name in the question, ONLY ask for clarification and stop.
PREDICTIONS: If asked about future events, acknowledge and analyze historical data instead.
RULES: Write 2-3 sentences of insight. If no ambiguity and more than 2 rows add:
CHART_SPEC:{{"label_col":"<col>","value_col":"<col>","type":"bar|line","title":"<title>"}}
Never mention SQL, markdown, or chart types in text."""

            interpret_response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({"prompt": f"<s>[INST]{interpret_prompt}[/INST]", "max_tokens": 400, "temperature": 0.3}),
                contentType="application/json", accept="application/json"
            )
            raw = json.loads(interpret_response["body"].read()).get("outputs", [{}])[0].get("text", "").strip()
            print(f"[chat] Interpret: {raw[:200]}")

            chart_json, insight_text = None, raw
            if "CHART_SPEC:" in raw:
                parts = raw.split("CHART_SPEC:", 1)
                insight_text = parts[0].strip()
                spec_raw = parts[1].strip().replace("```json","").replace("```","").strip()
                try:
                    spec = json.loads(spec_raw[:spec_raw.index("}")+1])
                    lc, vc = spec.get("label_col"), spec.get("value_col")
                    if lc in col_names and vc in col_names:
                        chart_json = {"type": spec.get("type","bar"), "title": spec.get("title", question[:60]),
                            "labels": [str(r.get(lc,"")) for r in results[:50]],
                            "datasets": [{"label": vc.replace("_"," ").title(), "data": [float(r.get(vc,0) or 0) for r in results[:50]], "color": "#378ADD"}]}
                except Exception as e:
                    print(f"[chat] CHART_SPEC error: {e}")

            if not insight_text:
                insight_text = f"Here are the results for: {contextual_question[:50]}."

            if any(s in insight_text.lower() for s in ["did you mean", "please specify", "multiple players", "clarify"]):
                chart_json = None
                q_idx = insight_text.find("?")
                if q_idx >= 0:
                    insight_text = insight_text[:q_idx+1].strip()

            full_response = f"{insight_text}\nCHART_JSON:{json.dumps(chart_json)}" if chart_json else insight_text
            return respond(200, {"content": [{"type": "text", "text": full_response}], "sql": sql_query, "results_count": len(results)})

        # GET /predictions?year=YYYY
        if path == "/predictions" and method == "GET":
            year = int(params.get("year", __import__("datetime").datetime.now().year))
            cur.execute("""
                SELECT
                    e.event_name, e.date::text, e.id_event::text,
                    mr.id_run::text, mr.model_version, mr.run_timestamp::text,
                    pl.tier, pl.predicted_total, pl.dg_proj_total, pl.salary_total,
                    pa.actual_total, pa.my_error, pa.dg_error,
                    json_agg(
                        json_build_object(
                            'golfer', SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1),
                            'dg_id', plp.dg_id, 'sim_mean', pg.sim_mean,
                            'made_cut_pct', pg.made_cut_pct, 'salary', pg.salary
                        ) ORDER BY pg.sim_mean DESC
                    ) AS picks
                FROM prediction_lineup pl
                JOIN model_run mr ON mr.id_run = pl.id_run
                JOIN event e ON e.id_event = mr.id_event
                JOIN prediction_lineup_pick plp ON plp.id_lineup = pl.id_lineup
                JOIN player p ON p.dg_id = plp.dg_id
                JOIN prediction_golfer pg ON pg.id_run = mr.id_run AND pg.dg_id = plp.dg_id
                LEFT JOIN prediction_accuracy pa ON pa.id_run = mr.id_run AND pa.tier = pl.tier
                WHERE e.calendar_year = %(year)s
                AND mr.run_timestamp = (
                    SELECT MAX(mr2.run_timestamp) FROM model_run mr2 WHERE mr2.id_event = mr.id_event
                )
                GROUP BY e.event_name, e.date, e.id_event, mr.id_run, mr.model_version,
                         mr.run_timestamp, pl.tier, pl.predicted_total, pl.dg_proj_total,
                         pl.salary_total, pa.actual_total, pa.my_error, pa.dg_error
                ORDER BY e.date DESC, pl.tier
            """, {"year": year})
            return respond(200, cur.fetchall())

        # GET /prediction-golfers?id_event=UUID
        if path == "/prediction-golfers" and method == "GET":
            id_event = params.get("id_event")
            if not id_event:
                return respond(400, {"error": "id_event required"})
            cur.execute("""
                SELECT
                    SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer,
                    pg.dg_id, pg.salary, pg.sim_mean, pg.sim_sd, pg.sim_p_low, pg.sim_p_high,
                    pg.made_cut_pct, pg.dg_proj, pg.dg_std_dev,
                    dt.total_pts AS actual_pts,
                    dt.fin_text
                FROM prediction_golfer pg
                JOIN player p ON p.dg_id = pg.dg_id
                LEFT JOIN dfs_total dt ON dt.dg_id = pg.dg_id AND dt.id_event = %(id_event)s
                WHERE pg.id_run = (
                    SELECT id_run FROM model_run WHERE id_event = %(id_event)s ORDER BY run_timestamp DESC LIMIT 1
                )
                ORDER BY pg.sim_mean DESC
            """, {"id_event": id_event})
            return respond(200, cur.fetchall())

        # GET /prediction-distributions?id_event=UUID&dg_ids=1,2,3
        # Returns dist_samples for specified golfers (lineup picks on load, individual on demand)
        if path == "/prediction-distributions" and method == "GET":
            id_event = params.get("id_event")
            dg_ids_param = params.get("dg_ids", "")
            if not id_event:
                return respond(400, {"error": "id_event required"})

            # Parse dg_ids - if not provided, return all lineup picks (up to 18)
            if dg_ids_param:
                dg_ids = [int(x) for x in dg_ids_param.split(",") if x.strip().isdigit()]
            else:
                # Default: all golfers in any lineup for this event
                cur.execute("""
                    SELECT DISTINCT plp.dg_id
                    FROM prediction_lineup_pick plp
                    JOIN prediction_lineup pl ON pl.id_lineup = plp.id_lineup
                    WHERE pl.id_run = (
                        SELECT id_run FROM model_run WHERE id_event = %(id_event)s ORDER BY run_timestamp DESC LIMIT 1
                    )
                """, {"id_event": id_event})
                dg_ids = [r['dg_id'] for r in cur.fetchall()]

            if not dg_ids:
                return respond(200, [])

            cur.execute("""
                SELECT SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer,
                    pg.dg_id, pg.sim_mean, pg.sim_sd, pg.sim_p_low, pg.sim_p_high,
                    pg.made_cut_pct, pg.dist_samples
                FROM prediction_golfer pg
                JOIN player p ON p.dg_id = pg.dg_id
                WHERE pg.id_run = (
                    SELECT id_run FROM model_run WHERE id_event = %(id_event)s ORDER BY run_timestamp DESC LIMIT 1
                )
                AND pg.dg_id = ANY(%(dg_ids)s)
                ORDER BY pg.sim_mean DESC
            """, {"id_event": id_event, "dg_ids": dg_ids})
            return respond(200, cur.fetchall())
        # GET /prediction-golfer-accuracy?year=YYYY&dg_id=INT
        if path == "/prediction-golfer-accuracy" and method == "GET":
            year = int(params.get("year", __import__("datetime").datetime.now().year))
            dg_id = params.get("dg_id")

            if dg_id:
                # Single golfer per-event history
                cur.execute("""
                    SELECT
                        SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer,
                        pg.dg_id,
                        e.event_name,
                        e.date::text,
                        pg.sim_mean,
                        pg.dg_proj,
                        dt.total_pts AS actual_pts,
                        dt.total_pts - pg.sim_mean AS my_error,
                        dt.total_pts - pg.dg_proj AS dg_error,
                        dt.fin_text
                    FROM prediction_golfer pg
                    JOIN model_run mr ON mr.id_run = pg.id_run
                    JOIN event e ON e.id_event = mr.id_event
                    JOIN player p ON p.dg_id = pg.dg_id
                    JOIN dfs_total dt ON dt.dg_id = pg.dg_id AND dt.id_event = mr.id_event
                    WHERE e.calendar_year = %(year)s
                    AND pg.dg_id = %(dg_id)s
                    AND mr.run_timestamp = (
                        SELECT MAX(mr2.run_timestamp) FROM model_run mr2 WHERE mr2.id_event = mr.id_event
                    )
                    ORDER BY e.date ASC
                """, {"year": year, "dg_id": int(dg_id)})
                return respond(200, cur.fetchall())
            else:
                # Aggregated per-golfer accuracy across all completed events
                cur.execute("""
                    WITH golfer_accuracy AS (
                        SELECT
                            SPLIT_PART(p.player_name,', ',2)||' '||SPLIT_PART(p.player_name,', ',1) AS golfer,
                            pg.dg_id,
                            COUNT(*) AS events_predicted,
                            ROUND(AVG(ABS(dt.total_pts - pg.sim_mean))::numeric, 2) AS my_mae,
                            ROUND(AVG(ABS(dt.total_pts - pg.dg_proj))::numeric, 2) AS dg_mae,
                            ROUND(AVG(dt.total_pts - pg.sim_mean)::numeric, 2) AS avg_error,
                            ROUND(SQRT(AVG(POWER(dt.total_pts - pg.sim_mean, 2)))::numeric, 2) AS my_rmse,
                            ROUND(SQRT(AVG(POWER(dt.total_pts - pg.dg_proj, 2)))::numeric, 2) AS dg_rmse,
                            ROUND(MIN(ABS(dt.total_pts - pg.sim_mean))::numeric, 2) AS best_error,
                            ROUND(MAX(ABS(dt.total_pts - pg.sim_mean))::numeric, 2) AS worst_error
                        FROM prediction_golfer pg
                        JOIN model_run mr ON mr.id_run = pg.id_run
                        JOIN event e ON e.id_event = mr.id_event
                        JOIN player p ON p.dg_id = pg.dg_id
                        JOIN dfs_total dt ON dt.dg_id = pg.dg_id AND dt.id_event = mr.id_event
                        WHERE e.calendar_year = %(year)s
                        AND mr.run_timestamp = (
                            SELECT MAX(mr2.run_timestamp) FROM model_run mr2 WHERE mr2.id_event = mr.id_event
                        )
                        GROUP BY p.player_name, pg.dg_id
                        HAVING COUNT(*) >= 1
                    )
                    SELECT * FROM golfer_accuracy
                    ORDER BY my_mae ASC
                """, {"year": year})
                rows = cur.fetchall()

                # Return top 10 best, top 10 worst, full list for search
                all_rows = rows
                best10 = rows[:10]
                worst10 = sorted(rows, key=lambda r: r['my_mae'] if r['my_mae'] else 0, reverse=True)[:10]

                return respond(200, {
                    "best": best10,
                    "worst": worst10,
                    "all": all_rows
                })
        if path == "/prediction-history" and method == "GET":
            year = int(params.get("year", __import__("datetime").datetime.now().year))
            cur.execute("""
                SELECT
                    e.event_name,
                    e.date::text,
                    e.id_event::text,
                    mr.model_version,
                    pl.tier,
                    pl.predicted_total,
                    pl.dg_proj_total,
                    pl.salary_total,
                    pa.actual_total,
                    pa.my_error,
                    pa.dg_error
                FROM prediction_lineup pl
                JOIN model_run mr ON mr.id_run = pl.id_run
                JOIN event e ON e.id_event = mr.id_event
                LEFT JOIN prediction_accuracy pa ON pa.id_run = mr.id_run AND pa.tier = pl.tier
                WHERE e.calendar_year = %(year)s
                AND pa.actual_total IS NOT NULL
                AND mr.run_timestamp = (
                    SELECT MAX(mr2.run_timestamp) FROM model_run mr2 WHERE mr2.id_event = mr.id_event
                )
                ORDER BY e.date ASC, pl.tier
            """, {"year": year})
            return respond(200, cur.fetchall())
        if path == "/upcoming-event" and method == "GET":
            secret_name = os.environ.get("SECRET_NAME")
            secrets = get_secret(secret_name)
            api_key = secrets.get("api_key")

            # Fetch field and salary data in parallel
            req_field = urllib.request.Request(
                f"https://feeds.datagolf.com/field-updates?tour=pga&file_format=json&key={api_key}",
                headers={"Content-Type": "application/json"},
                method="GET"
            )
            with urllib.request.urlopen(req_field, timeout=15) as r:
                field_data = json.loads(r.read().decode("utf-8"))

            # Fetch current FanDuel salaries
            salary_map = {}
            try:
                req_salary = urllib.request.Request(
                    f"https://feeds.datagolf.com/preds/fantasy-projection-defaults?tour=pga&site=fanduel&slate=main&file_format=json&key={api_key}",
                    headers={"Content-Type": "application/json"},
                    method="GET"
                )
                with urllib.request.urlopen(req_salary, timeout=15) as r:
                    salary_data = json.loads(r.read().decode("utf-8"))
                for p in salary_data.get("projections", []):
                    if p.get("dg_id") and p.get("salary"):
                        salary_map[p["dg_id"]] = int(p["salary"])
            except Exception:
                pass

            # Get dg_ids of players in the field
            field = field_data.get("field", [])
            dg_ids = [p["dg_id"] for p in field if p.get("dg_id")]
            event_name = field_data.get("event_name", "")

            # Query historical value at this event for players in the current field
            value_picks = []
            if dg_ids and event_name:
                cur.execute("""
                    SELECT
                        p.player_name,
                        p.dg_id,
                        COUNT(*) AS appearances,
                        ROUND(AVG(dt.total_pts)::numeric, 1) AS avg_pts,
                        ROUND(AVG(dt.salary)::numeric, 0) AS avg_salary,
                        ROUND((AVG(dt.total_pts) / NULLIF(AVG(dt.salary), 0) * 1000)::numeric, 2) AS value_ratio,
                        ROUND(MAX(dt.total_pts)::numeric, 1) AS best_pts
                    FROM dfs_total dt
                    JOIN player p ON p.dg_id = dt.dg_id
                    JOIN event e ON e.id_event = dt.id_event
                    WHERE dt.dg_id = ANY(%(dg_ids)s)
                    AND e.event_name ILIKE %(event_pattern)s
                    AND dt.salary > 0
                    AND dt.fin_text NOT IN ('WD', 'CUT')
                    GROUP BY p.player_name, p.dg_id
                    HAVING COUNT(*) >= 1
                    ORDER BY value_ratio DESC
                    LIMIT 50
                """, {"dg_ids": dg_ids, "event_pattern": f"%{event_name.split(' ')[0]}%"})
                rows = cur.fetchall()
                # Merge current salary from DataGolf
                value_picks = []
                for row in rows:
                    pick = dict(row) if hasattr(row, 'keys') else {
                        "player_name": row[0], "dg_id": row[1],
                        "appearances": row[2], "avg_pts": row[3],
                        "avg_salary": row[4], "value_ratio": row[5], "best_pts": row[6]
                    }
                    pick["current_salary"] = salary_map.get(pick["dg_id"])
                    value_picks.append(pick)

            field_data["value_picks"] = value_picks
            return respond(200, field_data)

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
