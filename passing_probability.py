# -*- coding: utf-8 -*-
import os, gzip, json, math, bisect, random, glob, time, sys
from collections import Counter, defaultdict

# ===================== パス設定（環境に合わせて） =====================
BASE        = r"C:\Users\nocch\Rits_ISE\senior\NBA-Player-Movements\data"
PASSES_DIR  = os.path.join(BASE, r"passing_prob_labels\tracking_passes")
PRETTY_DIR  = os.path.join(BASE, "pretty_jsons_gz")
OUT_DIR     = os.path.join(BASE, r"passing_prob_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== ハイパーパラメータ =====================
DEF_NEAR_FT      = 4.0      # パス経路に「近い」DFの距離閾値
NO_DEF_DIST_FT   = 30.0     # 近傍DFがいないときに入れる距離（平均/最小）
TRAIN_RATIO      = 0.50     # train/test 50/50（論文に合わせる）
SEED             = 7        # 再現性
MODEL_NAME       = "pp_lr_ols.candidate_paths.ALL_fulltake.json"  # 出力ファイル名

random.seed(SEED)

# ===================== ユーティリティ =====================
def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def hypot2d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dot(ax, ay, bx, by):
    return ax*bx + ay*ay if False else ax*bx + ay*by  # guard for typos

# 点Pから線分ABへの最短距離（線分外への射影は除外：t∈[0,1]のみ）
def dist_point_to_segment(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    abx, aby = bx-ax, by-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0.0:
        return math.hypot(px-ax, py-ay), None
    apx, apy = px-ax, py-ay
    t = (apx*abx + apy*aby) / ab2
    if t < 0.0 or t > 1.0:
        return float("inf"), t
    cx, cy = ax + t*abx, ay + t*aby
    return math.hypot(px-cx, py-cy), t

# pretty（tracking）をtsで引けるようにインデックス化
def index_pretty(game_id):
    pretty_path = os.path.join(PRETTY_DIR, f"{game_id}_pretty.json.gz")
    trk = load_json_gz(pretty_path)
    frames = []
    for ev in trk["events"]:
        for m in ev["moments"]:
            ts = int(m[1])
            period = int(m[0])
            clock = float(m[2])
            ents = m[5]
            ball = None
            players = []
            for row in ents:
                tid, pid = int(row[0]), int(row[1])
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                if pid == -1:
                    ball = (x, y, z)
                else:
                    players.append((tid, pid, x, y, z))
            if ball is None:
                continue
            frames.append({"ts": ts, "period": period, "clock": clock, "ball": ball, "players": players})
    frames.sort(key=lambda d: d["ts"])
    ts_list = [f["ts"] for f in frames]
    return frames, ts_list

# ts_target に最も近いフレームを返す（許容なし＝常にいちばん近い）
def get_frame_near_ts(frames, ts_list, ts_target):
    i = bisect.bisect_left(ts_list, ts_target)
    candidates = []
    if 0 <= i < len(frames): candidates.append(frames[i])
    if i-1 >= 0: candidates.append(frames[i-1])
    if i+1 < len(frames): candidates.append(frames[i+1])
    best = None
    best_dt = 10**9
    for fr in candidates:
        dt = abs(fr["ts"] - ts_target)
        if dt < best_dt:
            best = fr; best_dt = dt
    return best

def player_xy_in_frame(frame, pid):
    for (tid, p, x, y, _z) in frame["players"]:
        if p == pid:
            return (x, y)
    return None

# 特定の選手pidの(x,y)を「中心tsに最も近いフレーム」から取る
def find_player_xy_near_ts(frames, ts_list, pid, center_ts):
    # 近傍3候補だけ見れば十分（実測的にOK）
    i = bisect.bisect_left(ts_list, center_ts)
    cand_idx = [i]
    if i-1 >= 0: cand_idx.append(i-1)
    if i+1 < len(frames): cand_idx.append(i+1)
    best_xy = None
    best_dt = 10**9
    for k in cand_idx:
        if k < 0 or k >= len(frames): continue
        fr = frames[k]
        for (tid, p, x, y, _z) in fr["players"]:
            if p == pid:
                dt = abs(fr["ts"] - center_ts)
                if dt < best_dt:
                    best_dt = dt
                    best_xy = (x, y)
                break
    # まれに3候補で拾えない場合は全域スキャン（頻度は低い）
    if best_xy is None:
        for fr in frames:
            for (tid, p, x, y, _z) in fr["players"]:
                if p == pid:
                    dt = abs(fr["ts"] - center_ts)
                    if dt < best_dt:
                        best_dt = dt
                        best_xy = (x, y)
                    break
    return best_xy

# ===================== 特徴量計算（全採用：BH&受け手あれば採用） =====================
def compute_features_for_group_fulltake(game_id, pass_obj, frames, ts_list, skip_counter=None):
    """
    1つのパス（グループ）に対して、候補（1〜4）ぶんの特徴量とyを返す。
      - “candidates”は使わず、リリース時フレームの同チーム（パサー除く）を候補に再構成
      - 最小要件：受け手(y=1)の座標が到着近傍で見つかること（見つからなければスキップ）
      - 候補が1〜4名でも採用（y=1が必ず1名）
    特徴：
      F1: pass_dist_ft（リリース位置→候補到着位置の直線距離）
      F2: n_def_near（上の直線から4ft以内にいる相手DFの人数：到着時刻フレームで評価）
      F3: avg_def_dist_ft（近傍DFの平均距離；いなければ NO_DEF_DIST_FT）
      F4: min_def_dist_ft（近傍DFの最小距離；いなければ NO_DEF_DIST_FT）
    """
    rel_ts   = pass_obj["release_ts"]
    arr_ts   = pass_obj["arrival_ts"]
    team_id  = pass_obj["team_id"]
    passer_id = pass_obj["passer_id"]
    receiver_id = pass_obj["receiver_id"]

    # リリース座標（ファイルにあればそれを、なければリリース近傍から）
    passer_xy = tuple(pass_obj.get("passer_xy_at_release") or ())
    if not passer_xy or len(passer_xy) != 2:
        fr_rel = get_frame_near_ts(frames, ts_list, rel_ts)
        if fr_rel is None:
            if skip_counter is not None: skip_counter["NO_REL_FRAME"] += 1
            return None
        px = player_xy_in_frame(fr_rel, passer_id)
        if px is None:
            if skip_counter is not None: skip_counter["PASSER_XY_MISSING"] += 1
            return None
        passer_xy = px
    else:
        fr_rel = get_frame_near_ts(frames, ts_list, rel_ts)

    # 到着に最も近いフレーム
    fr_arr = get_frame_near_ts(frames, ts_list, arr_ts)
    if fr_arr is None:
        if skip_counter is not None: skip_counter["NO_ARR_FRAME"] += 1
        return None

    # 到着フレームの相手ディフェンダ
    defenders = [(p, (x, y)) for (tid, p, x, y, _z) in fr_arr["players"] if tid != team_id]

    # リリース時フレームで候補（同チーム、パサー以外）を再構成
    if fr_rel is None:
        fr_rel = get_frame_near_ts(frames, ts_list, rel_ts)
        if fr_rel is None:
            if skip_counter is not None: skip_counter["NO_REL_FRAME"] += 1
            return None
    team_candidates = [p for (tid, p, x, y, _z) in fr_rel["players"] if tid == team_id and p != passer_id]
    # まれに重複・抜けを掃除
    team_candidates = list(dict.fromkeys(team_candidates))
    if len(team_candidates) == 0:
        if skip_counter is not None: skip_counter["NO_TEAMMATES_AT_RELEASE"] += 1
        return None

    # 受け手は必須（見つからなければスキップ）
    rec_xy = find_player_xy_near_ts(frames, ts_list, receiver_id, arr_ts)
    if rec_xy is None:
        if skip_counter is not None: skip_counter["RECEIVER_XY_MISSING"] += 1
        return None

    # 内部ヘルパ：特徴量を作る
    def make_feature_row(pid, xy, is_receiver):
        pass_dist = hypot2d(passer_xy, xy)
        near_ds = []
        for _dpid, dxy in defenders:
            d_line, t = dist_point_to_segment(dxy, passer_xy, xy)
            if d_line <= DEF_NEAR_FT and t is not None and 0.0 <= t <= 1.0:
                near_ds.append(d_line)
        n_near = len(near_ds)
        if n_near == 0:
            avg_d = NO_DEF_DIST_FT
            min_d = NO_DEF_DIST_FT
        else:
            avg_d = sum(near_ds) / n_near
            min_d = min(near_ds)
        return {
            "group_id": f"{game_id}:{rel_ts}",
            "game_id": game_id,
            "release_ts": rel_ts,
            "arrival_ts": arr_ts,
            "passer_id": passer_id,
            "candidate_id": pid,
            "y": 1 if is_receiver else 0,
            "F": {
                "pass_dist_ft": pass_dist,
                "n_def_near": n_near,
                "avg_def_dist_ft": avg_d,
                "min_def_dist_ft": min_d,
            }
        }

    out_rows = []
    # まず受け手（必ず入れる）
    out_rows.append(make_feature_row(receiver_id, rec_xy, True))

    # ほかの候補（受け手以外、到着近傍座標が取れたぶんだけ）
    for pid in team_candidates:
        if pid == receiver_id:
            continue
        xy = find_player_xy_near_ts(frames, ts_list, pid, arr_ts)
        if xy is None:
            if skip_counter is not None: skip_counter["CAND_XY_MISSING"] += 1
            continue
        out_rows.append(make_feature_row(pid, xy, False))

    # 受け手は必ず1名
    if sum(r["y"] for r in out_rows) != 1:
        if skip_counter is not None: skip_counter["Y1_NOT_UNIQUE"] += 1
        return None

    return out_rows

# ===================== 進捗バー =====================
def print_progress(done, total, game_id, add_groups, add_rows, start_ts):
    bar_len = 30
    ratio = done / total if total else 1.0
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.time() - start_ts
    eta = (elapsed / ratio - elapsed) if ratio > 0 else 0.0
    sys.stdout.write(
        f"\r[{bar}] {ratio*100:5.1f}%  ({done}/{total})  {game_id}: +groups {add_groups}, +rows {add_rows} | elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s"
    )
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")

# ===================== データセット作成（全試合｜全採用） =====================
def collect_dataset_fulltake():
    dataset = []
    skip = Counter()
    files = sorted(glob.glob(os.path.join(PASSES_DIR, "*_passes.v1.json.gz")))
    print(f"検出 passes ファイル数: {len(files)}")
    t0 = time.time()
    for idx, pth in enumerate(files, 1):
        try:
            obj = load_json_gz(pth)
        except Exception:
            skip["READ_ERROR"] += 1
            print_progress(idx, len(files), os.path.basename(pth), 0, 0, t0)
            continue

        game_id = obj.get("game_id")
        labels = obj.get("labels", [])
        try:
            frames, ts_list = index_pretty(game_id)
        except Exception:
            skip["NO_PRETTY"] += 1
            print_progress(idx, len(files), game_id, 0, 0, t0)
            continue

        before_groups = 0
        before_rows = 0
        for p in labels:
            rows = compute_features_for_group_fulltake(game_id, p, frames, ts_list, skip_counter=skip)
            if rows is None:
                continue
            dataset.extend(rows)
            before_groups += 1
            before_rows += len(rows)

        print_progress(idx, len(files), game_id, before_groups, before_rows, t0)

    return dataset, skip

# ===================== 線形回帰（最小二乗：バイアス含む） =====================
def fit_ols(X, y):
    import numpy as np
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xb = np.hstack([ones, X])  # [c0, c1..c4]
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta  # len=5

def r2_score(y_true, y_pred):
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred)**2).sum())
    mean_y = float(y_true.mean()) if len(y_true) else 0.0
    ss_tot = float(((y_true - mean_y)**2).sum())
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

# グループTop-k（可変人数対応：k>人数なら人数に丸め）
def topk_by_groups(rows, yhat_by_row, k):
    bucket = defaultdict(list)
    for i, r in enumerate(rows):
        bucket[r["group_id"]].append(i)
    hit = 0; total = 0
    for gid, idxs in bucket.items():
        total += 1
        pos1 = [i for i in idxs if rows[i]["y"] == 1]
        if len(pos1) != 1:
            continue
        i1 = pos1[0]
        ranked = sorted(idxs, key=lambda i: yhat_by_row[i], reverse=True)
        kk = min(k, len(ranked))
        if i1 in set(ranked[:kk]):
            hit += 1
    return hit, total, (hit/total if total else 0.0)

# ===================== メイン処理 =====================
def main():
    # 1) データ収集（全試合｜全採用）
    dataset, skip = collect_dataset_fulltake()
    print("スキップ内訳:", dict(skip))
    if not dataset:
        print("有効サンプルがありません。前段の tracking_passes を確認してください。")
        return

    # 2) 行ごとにX,yを用意
    rows = dataset
    feats = ["pass_dist_ft", "n_def_near", "avg_def_dist_ft", "min_def_dist_ft"]
    X = [[r["F"][f] for f in feats] for r in rows]
    y = [r["y"] for r in rows]

    # 3) グループ単位で train/test split（グループIDで分割）
    groups = sorted(set(r["group_id"] for r in rows))
    random.shuffle(groups)
    n_train = int(len(groups) * TRAIN_RATIO)
    train_g = set(groups[:n_train])
    test_g  = set(groups[n_train:])

    train_idx = [i for i, r in enumerate(rows) if r["group_id"] in train_g]
    test_idx  = [i for i, r in enumerate(rows) if r["group_id"] in test_g]

    def take(lst, idxs): return [lst[i] for i in idxs]
    X_train, y_train = take(X, train_idx), take(y, train_idx)
    X_test,  y_test  = take(X, test_idx),  take(y, test_idx)

    # 4) OLS学習
    beta = fit_ols(X_train, y_train)
    c0, c1, c2, c3, c4 = [float(b) for b in beta]

    # 5) 予測（候補スコア）
    import numpy as np
    def predict_rows(Xlist):
        Xarr = np.asarray(Xlist, dtype=float)
        return c0 + Xarr[:,0]*c1 + Xarr[:,1]*c2 + Xarr[:,2]*c3 + Xarr[:,3]*c4

    yhat_train = predict_rows(X_train)
    yhat_test  = predict_rows(X_test)

    # 6) R^2（候補レベル）
    R2_train = r2_score(y_train, yhat_train)
    R2_test  = r2_score(y_test,  yhat_test)

    # 7) Top-k（グループ=各パス、可変候補人数対応）
    rows_train = [rows[i] for i in train_idx]
    rows_test  = [rows[i] for i in test_idx]

    hit1_tr, tot_tr, top1_tr = topk_by_groups(rows_train, yhat_train, k=1)
    hit2_tr, _,      top2_tr = topk_by_groups(rows_train, yhat_train, k=2)
    hit3_tr, _,      top3_tr = topk_by_groups(rows_train, yhat_train, k=3)

    hit1_te, tot_te, top1_te = topk_by_groups(rows_test,  yhat_test,  k=1)
    hit2_te, _,      top2_te = topk_by_groups(rows_test,  yhat_test,  k=2)
    hit3_te, _,      top3_te = topk_by_groups(rows_test,  yhat_test,  k=3)

    # 8) 出力
    print("\n=== Passing-Probability モデル（全試合・候補経路｜全採用=BH&受け手必須） ===")
    print(f"総サンプル（候補行） : {len(rows)}  （= 各パスの候補人数ぶん合算）")
    print(f"グループ数（総パス） : {len(groups)}")
    print(f"分割: train={len(train_g)} groups / test={len(test_g)} groups")
    print("\n--- 係数（y = c0 + c1*dist + c2*n_def + c3*avg_def + c4*min_def） ---")
    print(f"  c0={c0:.3f}, c1={c1:.3f}, c2={c2:.3f}, c3={c3:.3f}, c4={c4:.3f}")
    print("\n--- R^2（候補レベル） ---")
    print(f"  Train R^2 = {R2_train:.3f}")
    print(f"  Test  R^2 = {R2_test:.3f}")
    print("\n--- Top-k（受け手=Top-k内｜可変人数対応） ---")
    print(f"[Train] Top-1: {hit1_tr}/{tot_tr} ({top1_tr*100:.1f}%) | Top-2: {hit2_tr}/{tot_tr} ({top2_tr*100:.1f}%) | Top-3: {hit3_tr}/{tot_tr} ({top3_tr*100:.1f}%)")
    print(f"[Test ] Top-1: {hit1_te}/{tot_te} ({top1_te*100:.1f}%) | Top-2: {hit2_te}/{tot_te} ({top2_te*100:.1f}%) | Top-3: {hit3_te}/{tot_te} ({top3_te*100:.1f}%)")

    # 9) 保存（モデル+メトリクスのJSON） ※同名なら上書き
    out = {
        "version": "pp.lr_ols.candidate_paths.ALL.fulltake",
        "params": {
            "DEF_NEAR_FT": DEF_NEAR_FT,
            "NO_DEF_DIST_FT": NO_DEF_DIST_FT,
            "TRAIN_RATIO": TRAIN_RATIO,
            "SEED": SEED
        },
        "features": ["pass_dist_ft","n_def_near","avg_def_dist_ft","min_def_dist_ft"],
        "coefficients": {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4},
        "metrics": {
            "train": {"R2": R2_train, "topk": {"top1": hit1_tr, "top2": hit2_tr, "top3": hit3_tr, "total": tot_tr}},
            "test":  {"R2": R2_test,  "topk": {"top1": hit1_te, "top2": hit2_te, "top3": hit3_te, "total": tot_te}},
        },
        "data_summary": {
            "rows": len(rows),
            "groups": len(groups),
            "skip_reasons": dict(skip)
        },
        "sources": {
            "passes_dir": PASSES_DIR,
            "pretty_dir": PRETTY_DIR
        }
    }
    save_json(os.path.join(OUT_DIR, MODEL_NAME), out)
    print(f"\n[保存] {os.path.join(OUT_DIR, MODEL_NAME)}")

if __name__ == "__main__":
    main()
