# BH判定
# -*- coding: utf-8 -*-
# shoot_expect.py — 1試合だけ処理し、{gameid}_labels.json.gz を出力（整形JSON）
# 修正点:
#   (1) 受け手=BH の取り違えを検出して t★を無効化
#   (2) t★候補4人に必ず受け手を含める（同フレーム不在なら ±2フレームで補完）
#   (3) t★スナップショットのフォールバック（インデックス不正/欠損時は時刻最接近フレーム）
#
# 出力:
#   out/gz/{gameid}_labels.json.gz
#   端末サマリのみ（CSV等なし）

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque

# =========================
# デフォルト（引数なし実行時）
# =========================
BASE_DIR = Path("C:/Users/nocch/Rits_ISE/senior/NBA-Player-Movements")
DEFAULT_PBP_FILE    = BASE_DIR / "data" / "pbp_data_gz" / "0021500001.pbp.json.gz"
DEFAULT_PRETTY_FILE = BASE_DIR / "data" / "pretty_jsons_gz" / "0021500001_pretty.json.gz"
DEFAULT_OUT_DIR     = BASE_DIR / "out"

# =========================
# パラメータ（必要に応じ調整）
# =========================
SPEED_CUTTER      = 6.0   # カッター速度しきい値
SPEED_SCREENER    = 1.0   # スクリーナー（ほぼ静止）の速度しきい値
START_FRAMES      = 3     # 連続フレームで off-ball 開始
END_FRAMES        = 3     # 条件崩壊 or キャッチ連続フレームで終了

CATCH_DIST        = 4.0   # ボール保持判定距離
CATCH_FRAMES      = 2     # ボール近接の継続で保持
HOLD_FRAMES       = 2     # 受け手の安定保持（パス確定）
COOL_DOWN_FRAMES  = 5     # 同一(A→B)の連続検出を抑制

PASS_DEF_TH       = 8.5   # パスライン近傍とみなすDF距離（線分-点距離）

# =========================
# ユーティリティ
# =========================
def find_first_pair(pbp_dir: Path, pretty_dir: Path) -> Optional[Tuple[Path, Path]]:
    pbps = sorted(pbp_dir.glob("*.pbp.json.gz"))
    pretties = {p.name.replace("_pretty.json.gz", ""): p for p in sorted(pretty_dir.glob("*_pretty.json.gz"))}
    for p in pbps:
        gid = p.name.replace(".pbp.json.gz", "")
        q = pretties.get(gid)
        if q and p.exists() and q.exists():
            return p, q
    return None

def mmss_str(clock_sec: float) -> str:
    s = max(float(clock_sec), 0.0)
    m = int(s // 60)
    rem = s - m * 60
    if abs(rem - int(rem)) < 1e-6:
        return f"{m:02d}:{int(rem):02d}"
    return f"{m:02d}:{rem:04.1f}"

def pctimestr_to_seconds(pct: str) -> float:
    m, s = pct.split(":")
    return int(m) * 60 + int(s)

def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)

def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 0:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)

def nearest_index_by_clock(frames: List[Dict[str, Any]], target_clock: float) -> Optional[int]:
    if not frames:
        return None
    best_j = None
    best_diff = 1e18
    for j, fr in enumerate(frames):
        d = abs(float(fr["game_clock"]) - float(target_clock))
        if d < best_diff:
            best_diff = d
            best_j = j
    return best_j

# =========================
# ロード
# =========================
def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def load_pbp_events(path: Path) -> List[Dict[str, Any]]:
    data = load_json_gz(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "events" in data:
        return data["events"]
    raise ValueError("Unsupported PBP JSON: " + str(path))

def load_pretty_raw(path: Path) -> Dict[str, Any]:
    obj = load_json_gz(path)
    return obj if isinstance(obj, dict) else {"events": obj}

def extract_teams_and_roster(pretty_raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Any]]:
    teams: Dict[str, Any] = {}
    roster: Dict[int, Any] = {}
    for ev in pretty_raw.get("events", []):
        if isinstance(ev, dict) and ("visitor" in ev) and ("home" in ev):
            for side in ("home", "visitor"):
                t = ev[side]
                teamid = int(t.get("teamid"))
                name = t.get("name", "")
                abbr = t.get("abbreviation", "")
                teams[str(teamid)] = {"name": name, "abbreviation": abbr}
                for p in t.get("players", []):
                    pid = int(p.get("playerid"))
                    fullname = f"{p.get('firstname', '')} {p.get('lastname', '')}".strip()
                    roster[pid] = {"name": fullname, "teamid": teamid}
            break
    return teams, roster

def extract_frames(pretty_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for ev in pretty_raw.get("events", []):
        for m in ev.get("moments", []):
            if not isinstance(m, list) or len(m) < 6:
                continue
            period, unix_ms, game_clock, shot_clock, _unused, plist = m
            frames.append({
                "period": int(period),
                "unix_ms": int(unix_ms),
                "game_clock": float(game_clock),
                "shot_clock": None if shot_clock is None else float(shot_clock),
                "players": plist
            })
    frames.sort(key=lambda r: (r["period"], -r["game_clock"]))  # period昇順、時計降順（進行順）
    return frames

# =========================
# PBP → ポゼッション
# =========================
def get_event_core(e: Dict[str, Any]) -> Tuple[int, int, float, str]:
    typ = int(e.get("EVENTMSGTYPE"))
    period = int(e.get("PERIOD"))
    pct = e.get("PCTIMESTRING")
    clk = 0.0 if pct is None else pctimestr_to_seconds(pct)
    desc = (e.get("HOMEDESCRIPTION") or e.get("VISITORDESCRIPTION") or e.get("NEUTRALDESCRIPTION") or "")
    return typ, period, clk, desc

def team_of_event(e: Dict[str, Any]) -> Optional[int]:
    for k in ("PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"):
        v = e.get(k)
        if v is None:
            continue
        try:
            vi = int(v)
        except Exception:
            continue
        if vi > 0:
            return vi
    return None

def safe_team_ids_from_pbp(pbp_events: List[Dict[str, Any]]) -> List[int]:
    s = set()
    for e in pbp_events:
        for k in ("PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"):
            v = e.get(k)
            if v is None:
                continue
            try:
                vi = int(v)
            except Exception:
                continue
            if vi > 0:
                s.add(vi)
    return sorted(s)

def other_team(team: Optional[int], both: List[int]) -> Optional[int]:
    if team is None:
        return None
    for t in both:
        if t != team:
            return t
    return None

def build_possessions_from_pbp(pbp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    team_ids_all = safe_team_ids_from_pbp(pbp)
    possessions: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    pending_shot_team: Optional[int] = None

    def close_and_open(now_period: int, now_clock: float, end_idx: int, end_desc: str, next_offense_team: Optional[int]):
        nonlocal current
        if current is not None:
            current["end_clock"] = now_clock
            current["end_idx"] = end_idx
            current["end_desc"] = end_desc
            possessions.append(current)
        current = {
            "period": now_period,
            "start_clock": now_clock,
            "end_clock": None,
            "start_idx": end_idx,
            "end_idx": None,
            "start_desc": "auto after: " + end_desc,
            "end_desc": None,
            "offense_team": next_offense_team
        }

    for i, e in enumerate(pbp):
        typ, period, clock, _ = get_event_core(e)

        # Start of period
        if typ == 12 and "Start of" in (e.get("NEUTRALDESCRIPTION") or ""):
            if current is not None and current.get("end_clock") is None:
                current["end_clock"] = clock
                current["end_idx"] = i
                current["end_desc"] = "forced by Start"
                possessions.append(current)
            current = {
                "period": period,
                "start_clock": clock,
                "end_clock": None,
                "start_idx": i,
                "end_idx": None,
                "start_desc": e.get("NEUTRALDESCRIPTION") or "Start",
                "end_desc": None,
                "offense_team": None
            }
            pending_shot_team = None
            continue

        # End of period
        if typ == 12 and "End of" in (e.get("NEUTRALDESCRIPTION") or ""):
            if current is not None and current.get("end_clock") is None:
                current["end_clock"] = clock
                current["end_idx"] = i
                current["end_desc"] = e.get("NEUTRALDESCRIPTION") or "End"
                possessions.append(current)
                current = None
            pending_shot_team = None
            continue

        if current is None:
            continue

        # Miss → 次のリバウンド判定
        if typ == 2:
            pending_shot_team = team_of_event(e) or pending_shot_team
            continue

        # Rebound → ディフェンス側が取ったら終了
        if typ == 4:
            if pending_shot_team is not None:
                reb = team_of_event(e)
                if reb is not None and reb != pending_shot_team:
                    close_and_open(period, clock, i, "DefReb", next_offense_team=reb)
                pending_shot_team = None
            continue

        # Make / TO / Violation → 終了
        if typ in (1, 5, 7):
            end_team = team_of_event(e)
            nxt = other_team(end_team, team_ids_all) if end_team is not None else None
            close_and_open(period, clock, i, f"type={typ}", next_offense_team=nxt)
            pending_shot_team = None
            continue

    # 残りを 0:00 でクローズ
    if current is not None and current.get("end_clock") is None:
        current["end_clock"] = 0.0
        current["end_idx"] = len(pbp) - 1
        current["end_desc"] = "forced 0:00"
        possessions.append(current)

    # clockはダウンする（start > end のみ残す）
    return [p for p in possessions if p["start_clock"] > p["end_clock"]]

# =========================
# フレーム・速度
# =========================
def split_players(plist: List[List[float]]) -> Tuple[Optional[Tuple[float, float]], List[Dict[str, Any]]]:
    ball_xy: Optional[Tuple[float, float]] = None
    players: List[Dict[str, Any]] = []
    for ent in plist:
        team_id, pid, x, y, z = ent
        team_id = int(team_id)
        pid = int(pid)
        if team_id == -1 and pid == -1:
            ball_xy = (float(x), float(y))
        else:
            players.append({"team_id": team_id, "playerid": pid, "x": float(x), "y": float(y), "z": float(z)})
    return ball_xy, players

def build_speed_map(history: Dict[int, deque], players: List[Dict[str, Any]], t_ms: int, hist_len: int) -> Dict[int, float]:
    sp: Dict[int, float] = {}
    for p in players:
        pid = int(p["playerid"])
        h = history[pid]
        h.append((t_ms, float(p["x"]), float(p["y"])))
        if len(h) < 2:
            sp[pid] = 0.0
            continue
        dist_sum = 0.0
        t_sum = 0.0
        prev = None
        for itm in h:
            if prev is not None:
                dt = (itm[0] - prev[0]) / 1000.0
                if dt > 0:
                    dist_sum += dist(itm[1], itm[2], prev[1], prev[2])
                    t_sum += dt
            prev = itm
        sp[pid] = (dist_sum / t_sum) if t_sum > 0 else 0.0
    return sp

# =========================
# パス検出（t★ = リリース直前）
# =========================
def detect_pass_events(cand_frames: List[Dict[str, Any]],
                       *,
                       catch_dist: float = CATCH_DIST,
                       hold_frames: int = HOLD_FRAMES,
                       cool_down: int = COOL_DOWN_FRAMES) -> List[Dict[str, Any]]:
    """
    - 各フレームでボール最接近者＝holder
    - holder が A→B に切替 → pending = {from:A, to:B, i_change}
    - 以後、B が hold_frames 連続で保持できたら確定（t★ は i_change-1）
    - クールダウン：直近と同一ペア (A,B) が cool_down フレーム内なら棄却
    - event: {i_star, period, t_star_clock, t_star_ms, bh_id(=A), receiver_id(=B)}
    """
    events: List[Dict[str, Any]] = []

    def holder_at_frame(fr: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        ball_xy, players = split_players(fr["players"])
        if ball_xy is None or not players:
            return None, None
        holder = min(players, key=lambda p: dist(p["x"], p["y"], ball_xy[0], ball_xy[1]))
        if dist(holder["x"], holder["y"], ball_xy[0], ball_xy[1]) < catch_dist:
            return holder["playerid"], holder["team_id"]
        return None, None

    prev_id: Optional[int] = None
    prev_team: Optional[int] = None

    pending: Optional[Dict[str, Any]] = None  # {"from":A, "to":B, "team":team_id, "i_change":i, "stable":k}
    last_pass_frame = -10**9
    last_pair: Optional[Tuple[int, int]] = None

    for i, fr in enumerate(cand_frames):
        hid, hteam = holder_at_frame(fr)

        if hid is None or hteam is None:
            prev_id, prev_team = hid, hteam
            pending = None
            continue

        if prev_id is not None and hid == prev_id:
            if pending and pending.get("to") == hid and pending.get("team") == hteam:
                pending["stable"] += 1
                if pending["stable"] >= hold_frames:
                    pair = (int(pending["from"]), int(pending["to"]))
                    if not (last_pair == pair and (pending["i_change"] - 1 - last_pass_frame) < cool_down):
                        i_star = max(int(pending["i_change"]) - 1, 0)
                        fr_star = cand_frames[i_star]
                        events.append({
                            "i_star": i_star,
                            "period": int(fr_star["period"]),
                            "t_star_clock": float(fr_star["game_clock"]),
                            "t_star_ms": int(fr_star["unix_ms"]),
                            "bh_id": int(pending["from"]),
                            "receiver_id": int(pending["to"])
                        })
                        last_pair = pair
                        last_pass_frame = i_star
                    pending = None
            prev_id, prev_team = hid, hteam
            continue

        if prev_id is not None and prev_team is not None and hteam == prev_team:
            pending = {"from": int(prev_id), "to": int(hid), "team": int(hteam), "i_change": i, "stable": 1}
        else:
            pending = None

        prev_id, prev_team = hid, hteam

    return events

# =========================
# メイン
# =========================
def main() -> None:
    # 入力（デフォルト or 最初の一致ペア）
    pbp_path = DEFAULT_PBP_FILE
    pretty_path = DEFAULT_PRETTY_FILE
    out_dir = DEFAULT_OUT_DIR
    if (not pbp_path.exists()) or (not pretty_path.exists()):
        pair = find_first_pair(BASE_DIR / "data" / "pbp_data_gz", BASE_DIR / "data" / "pretty_jsons_gz")
        if pair:
            pbp_path, pretty_path = pair

    if not pbp_path.exists() or not pretty_path.exists():
        print("[ERROR] 入力ファイルが見つかりません。パスを直すか DEFAULT_* を修正してください。")
        print("  pbp   :", pbp_path)
        print("  pretty:", pretty_path)
        return

    (out_dir / "gz").mkdir(parents=True, exist_ok=True)

    # gameid
    gid = pbp_path.name.replace(".pbp.json.gz", "")
    if gid == pbp_path.name:
        gid = pretty_path.name.replace("_pretty.json.gz", "")

    # ロード
    pbp = load_pbp_events(pbp_path)
    pretty_raw = load_pretty_raw(pretty_path)
    frames = extract_frames(pretty_raw)
    teams, roster = extract_teams_and_roster(pretty_raw)

    # ポゼッション
    possessions = build_possessions_from_pbp(pbp)

    # periodごとのフレーム
    frames_by_period: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for fr in frames:
        frames_by_period[int(fr["period"])].append(fr)

    # 速度履歴
    hist_len = 3
    speed_hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=hist_len))

    out_possessions: List[Dict[str, Any]] = []
    total_intervals = 0
    total_tstar_events = 0
    per_counter: Dict[int, int] = defaultdict(int)

    for pos in possessions:
        period = int(pos["period"])
        per_counter[period] += 1
        pos_id = f"Q{period}-#{per_counter[period]}"
        start_clock = float(pos["start_clock"])
        end_clock = float(pos["end_clock"])

        # このポゼッションのフレーム
        cand = [fr for fr in frames if fr["period"] == period and (fr["game_clock"] <= start_clock + 1e-6) and (fr["game_clock"] > end_clock + 1e-6)]
        if not cand:
            continue

        # t★列挙（ポゼッション内のすべてのパス前フレーム）
        pass_events = detect_pass_events(
            cand,
            catch_dist=CATCH_DIST,
            hold_frames=HOLD_FRAMES,
            cool_down=COOL_DOWN_FRAMES
        )
        total_tstar_events += len(pass_events)
        tstar_by_index = {ev["i_star"]: ev for ev in pass_events}

        intervals_out: List[Dict[str, Any]] = []
        active: Dict[int, Dict[str, Any]] = {}
        consec_start: Dict[int, int] = defaultdict(int)
        consec_end: Dict[int, int] = defaultdict(int)
        consec_catch: Dict[int, int] = defaultdict(int)

        # 近傍でIDの座標を探すヘルパ（±windowフレーム）
        def find_player_in_window(idx: int, player_id: int, window: int = 2) -> Optional[Dict[str, Any]]:
            start = max(0, idx - window)
            end = min(len(cand) - 1, idx + window)
            for j in range(start, end + 1):
                bxy2, pls2 = split_players(cand[j]["players"])
                if bxy2 is None:
                    continue
                hit = next((pp for pp in pls2 if int(pp["playerid"]) == int(player_id)), None)
                if hit is not None:
                    return {
                        "playerid": int(hit["playerid"]),
                        "team_id": int(hit["team_id"]),
                        "x": float(hit["x"]),
                        "y": float(hit["y"]),
                        "z": float(hit["z"])
                    }
            return None

        # 区間走査
        for i, fr in enumerate(cand):
            clock = float(fr["game_clock"])
            t_ms = int(fr["unix_ms"])
            bxy, pls = split_players(fr["players"])
            if bxy is None or len(pls) != 10:
                continue

            # そのフレームのBHで攻撃側を定義
            bh_tmp = min(pls, key=lambda p: dist(p["x"], p["y"], bxy[0], bxy[1]))
            off_team_now = bh_tmp["team_id"]
            offense = [p for p in pls if p["team_id"] == off_team_now]
            defense = [p for p in pls if p["team_id"] != off_team_now]
            if not offense or len(defense) == 0:
                continue

            sp_map = build_speed_map(speed_hist, pls, t_ms, hist_len)
            bh = min(offense, key=lambda p: dist(p["x"], p["y"], bxy[0], bxy[1]))
            bh_id = bh["playerid"]
            off_right = (bh["x"] >= 47.0)

            # start 監視（BH以外）
            for p in offense:
                if p["playerid"] == bh_id:
                    continue
                pid = p["playerid"]
                spd = sp_map.get(pid, 0.0)
                in_half = (p["x"] >= 47.0) if off_right else (p["x"] < 47.0)
                cond = (spd >= SPEED_CUTTER) and in_half

                consec_start[pid] = consec_start[pid] + 1 if cond else 0
                if consec_start[pid] == START_FRAMES and pid not in active:
                    active[pid] = {"t_start_clock": clock, "t_start_ms": t_ms, "i_start": i}

                d_ball = dist(p["x"], p["y"], bxy[0], bxy[1])
                if cond:
                    consec_end[pid] = 0
                else:
                    consec_end[pid] += 1
                if d_ball < CATCH_DIST or pid == bh_id:
                    consec_catch[pid] += 1
                else:
                    consec_catch[pid] = 0

            # 終了条件
            for pid in list(active.keys()):
                pnow = next((pp for pp in offense if pp["playerid"] == pid), None)
                end_by_leave = (pnow is None)
                end_by_cond  = (consec_end[pid] >= END_FRAMES or consec_catch[pid] >= CATCH_FRAMES)
                if end_by_leave or end_by_cond:
                    st = active[pid]
                    t_start_clock = st["t_start_clock"]
                    i_start = st["i_start"]
                    i_end = i
                    t_end_clock = clock

                    # 区間内の t★ を1つ拾う（t_start_clock ≥ t★ ≥ t_end_clock）
                    t_star_ev: Optional[Dict[str, Any]] = None
                    for j in range(i_start, i_end + 1):
                        ev = tstar_by_index.get(j)
                        if ev and (t_start_clock >= ev["t_star_clock"] >= t_end_clock):
                            t_star_ev = ev
                            break

                    # (1) 受け手=BH なら t★を無効化（取り違え保険）
                    if t_star_ev and int(t_star_ev["receiver_id"]) == int(t_star_ev["bh_id"]):
                        t_star_ev = None

                    # エンティティ整形
                    def enrich_ent(p: Dict[str, Any]) -> Dict[str, Any]:
                        rid = int(p["playerid"])
                        info = roster.get(rid, {"name": "", "teamid": p["team_id"]})
                        team = teams.get(str(info["teamid"]), {})
                        return {
                            "playerid": rid,
                            "name": info.get("name", ""),
                            "teamid": info.get("teamid", p["team_id"]),
                            "team": team.get("abbreviation", ""),
                            "x": float(p["x"]),
                            "y": float(p["y"])
                        }

                    # スナップショット取得（t★はパサーIDを固定してBHに据える）
                    def snapshot_at(idx: Optional[int], tag: str, force_bh_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
                        if idx is None:
                            return None
                        if idx < 0 or idx >= len(cand):
                            return None
                        frx = cand[idx]
                        bxy2, pls2 = split_players(frx["players"])
                        if bxy2 is None:
                            return None
                        if force_bh_id is not None:
                            bh2 = next((pp for pp in pls2 if pp["playerid"] == force_bh_id), None)
                            if bh2 is None:
                                bh2 = min(pls2, key=lambda pp: dist(pp["x"], pp["y"], bxy2[0], bxy2[1]))
                        else:
                            bh2 = min(pls2, key=lambda pp: dist(pp["x"], pp["y"], bxy2[0], bxy2[1]))
                        off_team2 = bh2["team_id"]
                        offense2 = [pp for pp in pls2 if pp["team_id"] == off_team2]
                        defense2 = [pp for pp in pls2 if pp["team_id"] != off_team2]
                        if not offense2:
                            return None
                        # このスナップショットの「対象カッター」は区間pid
                        cutter2 = next((pp for pp in offense2 if pp["playerid"] == pid), bh2)

                        # スクリーン近傍抽出
                        sp2 = build_speed_map(speed_hist, pls2, int(frx["unix_ms"]), hist_len)
                        screeners2: List[Dict[str, Any]] = []
                        for spp in offense2:
                            if spp["playerid"] in (bh2["playerid"], cutter2["playerid"]):
                                continue
                            if sp2.get(spp["playerid"], 0.0) < SPEED_SCREENER and dist(
                                spp["x"], spp["y"], cutter2["x"], cutter2["y"]
                            ) < 3.0:
                                screeners2.append(spp)
                        other_off2 = [
                            spp for spp in offense2
                            if spp["playerid"] not in (bh2["playerid"], cutter2["playerid"])
                            and all(spp["playerid"] != s["playerid"] for s in screeners2)
                        ]

                        return {
                            "time": {
                                "pos_id": pos_id,
                                "tag": tag,
                                "period": int(frx["period"]),
                                "clock_sec": float(frx["game_clock"]),
                                "clock_mmss": mmss_str(float(frx["game_clock"])),
                                "unix_ms": int(frx["unix_ms"]),
                                "abs_game_sec": (int(frx["period"]) - 1) * 720.0 + (720.0 - float(frx["game_clock"]))
                            },
                            "ball": {"x": float(bxy2[0]), "y": float(bxy2[1])},
                            "bh": enrich_ent(bh2),
                            "cutter": enrich_ent(cutter2),
                            "screeners": [enrich_ent(s) for s in screeners2],
                            "other_offense": [enrich_ent(o) for o in other_off2],
                            "defense": [enrich_ent(d) for d in defense2]
                        }

                    i_star = t_star_ev["i_star"] if t_star_ev else None

                    # (3) t★スナップショットのフォールバック
                    if t_star_ev and (i_star is None or not (0 <= i_star < len(cand))):
                        guess = nearest_index_by_clock(cand, float(t_star_ev["t_star_clock"]))
                        i_star = guess

                    snap = {
                        "t_start": snapshot_at(i_start, "t_start"),
                        "t_star": snapshot_at(i_star, "t_star", force_bh_id=(t_star_ev["bh_id"] if t_star_ev else None)) if i_star is not None else None,
                        "t_end": snapshot_at(i_end, "t_end")
                    }

                    # 候補4人（t★は「パス前BH基準」で BH以外の味方4人＋label）
                    def build_features_for_pair(bh2: Dict[str, Any], candp: Dict[str, Any], defense_list: List[Dict[str, Any]]) -> Dict[str, Any]:
                        dists: List[float] = []
                        for dply in defense_list:
                            dseg = point_to_segment_distance(dply["x"], dply["y"], bh2["x"], bh2["y"], candp["x"], candp["y"])
                            if dseg <= PASS_DEF_TH:
                                dists.append(dseg)
                        return {
                            "pass_distance": float(dist(bh2["x"], bh2["y"], candp["x"], candp["y"])),
                            "defenders_near": int(len(dists)),
                            "def_mean_dist": (float(sum(dists) / len(dists)) if dists else None),
                            "def_min_dist": (float(min(dists)) if dists else None)
                        }

                    def candidates_at_tstar(idx: Optional[int], ev: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
                        if idx is None or ev is None:
                            return []
                        if idx < 0 or idx >= len(cand):
                            return []
                        frx = cand[idx]
                        bxy2, pls2 = split_players(frx["players"])
                        if bxy2 is None:
                            return []
                        # BHはパサー固定（見つからなければ最接近者で代用）
                        bh2 = next((pp for pp in pls2 if int(pp["playerid"]) == int(ev["bh_id"])), None)
                        if bh2 is None:
                            bh2 = min(pls2, key=lambda pp: dist(pp["x"], pp["y"], bxy2[0], bxy2[1]))
                        off_team2 = int(bh2["team_id"])
                        offense2 = [pp for pp in pls2 if int(pp["team_id"]) == off_team2]
                        others = [pp for pp in offense2 if int(pp["playerid"]) != int(bh2["playerid"])]
                        defense2 = [pp for pp in pls2 if int(pp["team_id"]) != off_team2]
                        receiver_id = int(ev["receiver_id"])

                        # 距離の近い順で暫定4人
                        others.sort(key=lambda q: dist(bh2["x"], bh2["y"], q["x"], q["y"]))
                        cand_list = others[:4]

                        # 受け手を同フレームで探す
                        recv_same = next((p for p in others if int(p["playerid"]) == receiver_id), None)
                        recv_pp: Optional[Dict[str, Any]] = recv_same

                        # 見つからなければ ±2 フレームで補完
                        if recv_pp is None:
                            recv_pp = find_player_in_window(idx, receiver_id, window=2)

                        # フォースインクルード：受け手を必ず cand_list に入れる
                        if recv_pp is not None:
                            cand_ids = [int(p["playerid"]) for p in cand_list]
                            if receiver_id not in cand_ids:
                                if len(cand_list) < 4:
                                    cand_list.append(recv_pp)
                                else:
                                    # BHから最も遠い候補を受け手に差し替え
                                    far_idx = max(range(len(cand_list)),
                                                  key=lambda k: dist(bh2["x"], bh2["y"], cand_list[k]["x"], cand_list[k]["y"]))
                                    cand_list[far_idx] = recv_pp

                        out_list: List[Dict[str, Any]] = []
                        for candp in cand_list:
                            fid = int(candp["playerid"])
                            role = "Cutter" if fid == pid else "OffenseOther"
                            feat = build_features_for_pair(bh2, candp, defense2)
                            out_list.append({
                                "candidate_id": fid,
                                "candidate_role": role,
                                "bh_xy": [float(bh2["x"]), float(bh2["y"])],
                                "candidate_xy": [float(candp["x"]), float(candp["y"])],
                                "ball_xy": [float(bxy2[0]), float(bxy2[1])],
                                "features": feat,
                                "label": 1 if fid == receiver_id else 0
                            })
                        return out_list

                    def candidates_at(idx: int) -> List[Dict[str, Any]]:
                        if idx < 0 or idx >= len(cand):
                            return []
                        frx = cand[idx]
                        bxy2, pls2 = split_players(frx["players"])
                        if bxy2 is None:
                            return []
                        bh2 = min(pls2, key=lambda pp: dist(pp["x"], pp["y"], bxy2[0], bxy2[1]))
                        off_team2 = bh2["team_id"]
                        offense2 = [pp for pp in pls2 if pp["team_id"] == off_team2]
                        others = [pp for pp in offense2 if pp["playerid"] != bh2["playerid"]]
                        defense2 = [pp for pp in pls2 if pp["team_id"] != off_team2]
                        others.sort(key=lambda q: dist(bh2["x"], bh2["y"], q["x"], q["y"]))
                        cand_list = others[:4]

                        out_list: List[Dict[str, Any]] = []
                        for candp in cand_list:
                            fid = int(candp["playerid"])
                            role = "Cutter" if fid == pid else "OffenseOther"
                            feat = build_features_for_pair(bh2, candp, defense2)
                            out_list.append({
                                "candidate_id": fid,
                                "candidate_role": role,
                                "bh_xy": [float(bh2["x"]), float(bh2["y"])],
                                "candidate_xy": [float(candp["x"]), float(candp["y"])],
                                "ball_xy": [float(bxy2[0]), float(bxy2[1])],
                                "features": feat
                            })
                        return out_list

                    # t★が残っていれば候補生成、なければ空リスト
                    cand_t_star = candidates_at_tstar(i_star, t_star_ev) if t_star_ev else []

                    intervals_out.append({
                        "playerid": int(pid),
                        "t_start_clock": float(t_start_clock),
                        "t_star_clock": (None if not t_star_ev else float(t_star_ev["t_star_clock"])),
                        "t_end_clock": float(t_end_clock),
                        "snapshots": snap,
                        "candidates": {
                            "t_start": candidates_at(i_start),
                            "t_star": cand_t_star,
                            "t_end": candidates_at(i_end)
                        },
                        "labels": (None if not t_star_ev else {"t_star": {"receiver_id": int(t_star_ev["receiver_id"])}})
                    })
                    del active[pid]

        if not intervals_out:
            continue

        out_possessions.append({
            "pos_id": pos_id,
            "period": period,
            "start_clock": start_clock,
            "end_clock": end_clock,
            "start_mmss": mmss_str(start_clock),
            "end_mmss": mmss_str(end_clock),
            "intervals": intervals_out
        })
        total_intervals += len(intervals_out)

    out_obj = {
        "gameid": gid,
        "input_files": {"pbp": str(pbp_path), "pretty": str(pretty_path)},
        "params": {
            "speed_cutter": SPEED_CUTTER,
            "speed_screener": SPEED_SCREENER,
            "start_frames": START_FRAMES,
            "end_frames": END_FRAMES,
            "catch_dist": CATCH_DIST,
            "catch_frames": CATCH_FRAMES,
            "pass_def_th": PASS_DEF_TH,
            "hold_frames": HOLD_FRAMES,
            "cool_down_frames": COOL_DOWN_FRAMES
        },
        "teams": teams,
        "roster": {str(k): v for (k, v) in roster.items()},
        "counts": {
            "possessions": len(out_possessions),
            "intervals": total_intervals,
            "t_star_events": total_tstar_events
        },
        "possessions": out_possessions
    }

    out_gz = (out_dir / "gz" / f"{gid}_labels.json.gz")
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_gz, "wt", encoding="utf-8") as gz:
        json.dump(out_obj, gz, ensure_ascii=False, indent=2)

    # 端末サマリのみ
    print("=== DONE (1 game) ===")
    print(f"gameid             : {gid}")
    print(f"possessions(out)   : {len(out_possessions)}")
    print(f"off-ball intervals : {total_intervals}")
    print(f"t_star events      : {total_tstar_events}")
    print(f"output             : {out_gz}")

if __name__ == "__main__":
    main()
