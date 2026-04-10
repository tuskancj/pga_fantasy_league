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
            context  = body.get("context", {})

            if not messages:
                return respond(400, {"error": "messages required"})

            secrets = get_secret(os.environ.get("SECRET_NAME"))
            anthropic_key = secrets.get("anthropic_key")
            if not anthropic_key:
                return respond(500, {"error": "anthropic_key not found in secret"})

            # Extract the latest user question
            question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            print(f"[chat] Question: {question}")

            # Step 1: Call pga-vanna Lambda to generate SQL
            print(f"[chat] Calling pga-vanna...")
            lambda_client = boto3.client("lambda", region_name="us-east-1", config=__import__("botocore").config.Config(read_timeout=110, connect_timeout=10))
            vanna_response = lambda_client.invoke(
                FunctionName="pga-vanna",
                InvocationType="RequestResponse",
                Payload=json.dumps({"httpMethod": "POST", "body": json.dumps({"question": question})})
            )
            vanna_payload = json.loads(vanna_response["Payload"].read())
            vanna_body = json.loads(vanna_payload.get("body", "{}"))
            sql_query = vanna_body.get("sql", "").strip()
            print(f"[chat] pga-vanna returned SQL: {sql_query[:200]}")

            if not sql_query or "error" in vanna_body:
                return respond(200, {
                    "content": [{"type": "text", "text": "I'm having trouble with that question. Could you try rephrasing it or be more specific?"}],
                    "sql": None,
                    "results_count": 0
                })

            # Step 2: Execute SQL
            results = []
            col_names = []
            sql_failed = False
            try:
                sql_cur = conn.cursor()
                sql_cur.execute(sql_query)
                col_names = [desc[0] for desc in sql_cur.description]
                rows = sql_cur.fetchall()
                results = [dict(zip(col_names, row)) for row in rows]
                sql_cur.close()
            except Exception as sql_err:
                try: sql_cur.close()
                except: pass
                conn.rollback()
                sql_failed = True
                sql_query = f"SQL_ERROR: {str(sql_err)}"

            # Log every query to training_log as unapproved for review
            try:
                log_cur = conn.cursor()
                log_cur.execute(
                    "INSERT INTO training_log (question, sql_query, approved) VALUES (%s, %s, false)",
                    (question, sql_query)
                )
                conn.commit()
                log_cur.close()
            except Exception:
                conn.rollback()

            if sql_failed:
                return respond(200, {
                    "content": [{"type": "text", "text": "I'm having trouble with that question. Could you try rephrasing it or be more specific?"}],
                    "sql": sql_query,
                    "results_count": 0
                })

            # Step 3: Claude interprets results using question context and specifies the chart
            results_str = json.dumps(results[:75], cls=CustomEncoder)
            col_summary = ', '.join(col_names) if col_names else ''

            interpret_system = f"""You are a data analyst for a PGA FanDuel fantasy golf league.

The user asked: "{question}"

SQL result columns: {col_summary}
Results ({len(results)} rows):
{results_str}

Your job:
1. Write 1-2 sentences of insight in plain conversational English
2. If there are more than 2 rows, specify a chart using CHART_SPEC on its own line

CHART_SPEC format:
CHART_SPEC:{{"label_col":"<column for x-axis>","value_col":"<column for bar/line values>","type":"bar|line","title":"<chart title>"}}

Column selection rules:
- For year-over-year queries: label_col = calendar_year
- For manager comparisons: label_col = username
- For golfer comparisons: label_col = golfer (or player_name)
- For scoring questions: value_col = total_pts or avg_pts
- For value/ratio questions: value_col = value_ratio
- For pick counts: value_col = times_picked or times_selected
- For missed events: value_col = missed_events
- For wins: value_col = wins
- Never use calendar_year, event_id, or dg_id as value_col

NEVER use markdown, tables, code blocks or backticks.
NEVER mention SQL or databases."""

            payload = json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 400,
                "system": interpret_system,
                "messages": messages
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"Content-Type": "application/json", "x-api-key": anthropic_key, "anthropic-version": "2023-06-01"},
                method="POST"
            )
            with urllib.request.urlopen(req) as r:
                answer_response = json.loads(r.read().decode("utf-8"))

            raw = answer_response["content"][0]["text"].strip()

            # Parse CHART_SPEC and build chart directly from results — Lambda does the math
            chart_json = None
            insight_text = raw
            if "CHART_SPEC:" in raw:
                parts = raw.split("CHART_SPEC:")
                insight_text = parts[0].strip()
                try:
                    spec = json.loads(parts[1].strip())
                    label_col = spec.get("label_col")
                    value_col = spec.get("value_col")
                    chart_type = spec.get("type", "bar")
                    chart_title = spec.get("title", question[:60])
                    if label_col in col_names and value_col in col_names:
                        chart_json = {
                            "type": chart_type,
                            "title": chart_title,
                            "labels": [str(r.get(label_col, '')) for r in results[:50]],
                            "datasets": [{
                                "label": value_col.replace('_', ' ').title(),
                                "data": [float(r.get(value_col, 0) or 0) for r in results[:50]],
                                "color": "#1D9E75"
                            }]
                        }
                except Exception:
                    pass

            full_response = f"{insight_text}\nCHART_JSON:{json.dumps(chart_json)}" if chart_json else insight_text

            return respond(200, {
                "content": [{"type": "text", "text": full_response}],
                "sql": sql_query,
                "results_count": len(results)
            })

        # GET /upcoming-event
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
