"""
export_play_frames.py

Reads the frame-level tracking CSV and exports a single play's data
(player positions per frame + precomputed control-area geometry) as a
JSON file the Vue/D3 dashboard can load directly.

Usage:
    python export_play_frames.py tracking_data.csv 2025090400 648 play_648.json
"""

import sys
import json
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from scipy.spatial import ConvexHull

# Position groups that make up each toggleable "control area" layer.
# Add/remove groups here -- the dashboard just renders whatever keys exist.
CONTROL_AREA_GROUPS = {
    'defense_overall': ('DL', 'LB', 'DB'),
    'front':           ('DL', 'LB'),
    'secondary':       ('DB',),
    'offense_overall': ('QB', 'WR', 'RB', 'TE'),   # OL excluded -- see note in chat
    'skill_position':  ('WR', 'RB', 'TE'),
    'receivers':       ('WR',),
    'backfield':       ('RB', 'TE'),
}

RADIUS_YARDS = 2  # buffer distance (yards) expanded outward from the group's hull edge


def transform(direction_mult, x, y):
    """Same convention used in plot_formation: width on x-axis, downfield on y-axis, flipped."""
    px = -direction_mult * y
    py = direction_mult * x
    return px, py


def control_area_polygons(points, buffer_dist=RADIUS_YARDS):
    """Convex hull over all points in a group, buffered outward -- fills the
    space between players rather than leaving gaps between individual circles.
    Falls back to a simple buffered point/line if there aren't enough points
    for a hull (need >= 3 non-collinear points)."""
    if not points:
        return []

    if len(points) < 3:
        # Not enough points for a hull -- buffer whatever we have directly
        # (a single point becomes a circle, two points become a capsule shape).
        if len(points) == 1:
            shape = Point(points[0]).buffer(buffer_dist)
        else:
            shape = LineString(points).buffer(buffer_dist)
        geoms = shape.geoms if shape.geom_type == 'MultiPolygon' else [shape]
        return [[[round(c[0], 1), round(c[1], 1)] for c in g.exterior.coords] for g in geoms]

    points_arr = np.array(points)
    try:
        hull = ConvexHull(points_arr)
    except Exception:
        # degenerate (collinear) points -- fall back to line buffer
        shape = LineString(points).buffer(buffer_dist)
        geoms = shape.geoms if shape.geom_type == 'MultiPolygon' else [shape]
        return [[[round(c[0], 1), round(c[1], 1)] for c in g.exterior.coords] for g in geoms]

    hull_pts = points_arr[hull.vertices]
    buffered = ShapelyPolygon(hull_pts).buffer(buffer_dist)
    geoms = buffered.geoms if buffered.geom_type == 'MultiPolygon' else [buffered]
    return [[[round(c[0], 1), round(c[1], 1)] for c in g.simplify(0.2).exterior.coords] for g in geoms]


# ---------------------------------------------------------------------------
# Route break detection (adapted from Ben's detect_route_break) + per-player
# route profile export: full trajectory, speed/accel over time, angle-change
# series, and detected break points -- everything the dashboard needs to
# render a per-player route chart when that player is clicked.
# ---------------------------------------------------------------------------

ROUTE_SIDE_GROUPS = ('QB', 'RB', 'TE', 'WR')  # who gets route-break analysis


def detect_route_break(df_player, min_angle_change=5, window=5, group_gap=10):
    """df_player must be sorted by frame_id, one player's rows for the whole play.
    Returns (df_player w/ angle_change_rate column, peaks array, break_frames list,
    break_indices list) -- break_frames is [(frame_id, peak_idx, total_turn), ...]."""
    df_player = df_player.sort_values('frame_id').reset_index(drop=True)

    if len(df_player) < window or df_player['dir'].isna().all():
        df_player['angle_change_rate'] = np.nan
        return df_player, np.array([], dtype=int), [], []

    dir_rad = np.radians(df_player['dir'].ffill().bfill())
    sin_mean = np.sin(dir_rad).rolling(window, center=True).mean()
    cos_mean = np.cos(dir_rad).rolling(window, center=True).mean()
    smoothed_dir = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    smoothed_dir = smoothed_dir.bfill().ffill()

    angle_diff = np.abs(np.diff(smoothed_dir, prepend=smoothed_dir.iloc[0]))
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    df_player['angle_change_rate'] = angle_diff

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df_player['angle_change_rate'], height=1, distance=1)

    direction_mult = -1 if df_player['play_direction'].iloc[0] == 'left' else 1
    los_x_raw = df_player['x_ball_at_snap'].iloc[0]
    los_x = los_x_raw + 1 if df_player['play_direction'].iloc[0] == 'left' else los_x_raw - 1

    candidates = []
    for peak in peaks:
        start_idx = max(0, peak - 1)
        end_idx = min(len(df_player) - 1, peak + 1)
        total_turn = np.sum(df_player['angle_change_rate'].iloc[start_idx:end_idx + 1])
        peak_frame = int(df_player['frame_id'].iloc[peak])
        peak_x = df_player['x'].iloc[peak]

        downfield_dist = direction_mult * (peak_x - los_x)
        if downfield_dist < 0:
            continue  # behind the line of scrimmage -- not a route break

        if total_turn >= min_angle_change:
            candidates.append({'peak': int(peak), 'frame': peak_frame, 'total_turn': float(total_turn)})

    break_frames = []
    break_indices = []
    current_group = []
    for cand in candidates:
        if not current_group or (cand['peak'] - current_group[-1]['peak']) <= group_gap:
            current_group.append(cand)
        else:
            best = max(current_group, key=lambda c: c['total_turn'])
            break_frames.append((best['frame'], best['peak'], round(best['total_turn'], 1)))
            break_indices.append(best['peak'])
            current_group = [cand]
    if current_group:
        best = max(current_group, key=lambda c: c['total_turn'])
        break_frames.append((best['frame'], best['peak'], round(best['total_turn'], 1)))
        break_indices.append(best['peak'])

    return df_player, peaks, break_frames, break_indices


def build_player_route_profiles(play_df, direction_mult, ball_x_snap, ball_y_snap):
    """One entry per offensive skill player: full trajectory (in the same
    ball-relative plot coordinates as the frame animation), speed/accel over
    time, angle-change series, detected breaks, and core reception stats."""
    profiles = {}

    skill_ids = play_df.loc[
        play_df['position_group'].isin(ROUTE_SIDE_GROUPS) & (play_df['offense_play'] == True),
        'nfl_id'
    ].unique()

    for nfl_id in skill_ids:
        player_df = play_df[play_df['nfl_id'] == nfl_id].sort_values('frame_id').reset_index(drop=True)
        if player_df.empty:
            continue

        player_df, peaks, break_frames, break_indices = detect_route_break(player_df.copy())

        route_frames, speed_series, accel_series, angle_series = [], [], [], []
        total_distance = 0.0
        prev_px = prev_py = None

        for _, row in player_df.iterrows():
            rel_x = row['x'] - ball_x_snap
            rel_y = row['y'] - ball_y_snap
            px, py = transform(direction_mult, rel_x, rel_y)
            fid = int(row['frame_id'])

            route_frames.append([fid, round(px, 2), round(py, 2)])
            if row.get('s') is not None and not pd.isna(row.get('s')):
                speed_series.append([fid, round(float(row['s']), 2)])
            if row.get('a') is not None and not pd.isna(row.get('a')):
                accel_series.append([fid, round(float(row['a']), 2)])
            rate = row.get('angle_change_rate')
            if rate is not None and not pd.isna(rate):
                angle_series.append([fid, round(float(rate), 2)])

            if prev_px is not None:
                total_distance += float(np.hypot(px - prev_px, py - prev_py))
            prev_px, prev_py = px, py

        peak_frames_all = [int(player_df['frame_id'].iloc[p]) for p in peaks]
        break_frame_set = {b[0] for b in break_frames}

        first_row = player_df.iloc[0]

        stats = {
            'rec_yards': first_row.get('rec_yards') if not pd.isna(first_row.get('rec_yards')) else None,
            'rush_yards': first_row.get('rush_yards') if not pd.isna(first_row.get('rush_yards')) else None,
            'route_distance_yards': round(total_distance, 1),
            'max_speed': round(max((s[1] for s in speed_series), default=0), 2),
            'avg_speed': round(float(np.mean([s[1] for s in speed_series])), 2) if speed_series else None,
        }

        profiles[int(nfl_id)] = {
            'nfl_id': int(nfl_id),
            'name': first_row.get('player_name', ''),
            'position_group': first_row.get('position_group', ''),
            'route': route_frames,
            'speed': speed_series,
            'accel': accel_series,
            'angle_change_rate': angle_series,
            'peaks_all': peak_frames_all,
            'breaks': [{'frame_id': f, 'total_turn': t} for f, _, t in break_frames],
            'break_frame_set': sorted(break_frame_set),
            'stats': stats,
        }

    return profiles


def build_play_json(df, game_id, gsis_play_id):
    play_df = df[(df['game_id'] == game_id) & (df['gsis_play_id'] == gsis_play_id)].copy()
    if play_df.empty:
        raise ValueError(f"No rows found for game {game_id}, play {gsis_play_id}")

    play_df = play_df.sort_values(['frame_id', 'nfl_id'])

    first_row = play_df.iloc[0]
    direction_mult = -1 if first_row.get('play_direction') == 'left' else 1

    meta = {
        'game_id': int(game_id),
        'gsis_play_id': int(gsis_play_id),
        'play_description': first_row.get('play_description_with_jersey_numbers', ''),
        'offense_formation': first_row.get('offense_formation', ''),
        'team_coverage_type': first_row.get('team_coverage_type', ''),
        'success': bool(first_row.get('successful_play', False)),
        'big_play': bool(first_row.get('big_play', False)),
    }

    ball_x_snap = first_row.get('x_ball_at_snap')
    ball_y_snap = first_row.get('y_ball_at_snap')

    frames = []
    for frame_id, frame_df in play_df.groupby('frame_id'):
        players = []
        group_points = {key: [] for key in CONTROL_AREA_GROUPS}
        avg_points = {}

        for _, row in frame_df.iterrows():

            rel_x = row['x'] - ball_x_snap
            rel_y = row['y'] - ball_y_snap
            px, py = transform(direction_mult, rel_x, rel_y)

            pos_group = row['position_group']
            side = 'offense' if row.get('offense_play') else 'defense'

            players.append({
                'nfl_id': row['nfl_id'],
                'name': row.get('player_name', ''),
                'position_group': pos_group,
                'side': side,
                'x': round(px, 2),
                'y': round(py, 2),
                's': row.get('s'),
                'dir': row.get('dir'),
            })

            for key, groups in CONTROL_AREA_GROUPS.items():
                if pos_group in groups:
                    group_points[key].append((px, py))

        control_areas = {
            key: control_area_polygons(points)
            for key, points in group_points.items()
            if points
        }

        # simple average marker per group (mean of the same point sets)
        averages = {
            key: [round(float(np.mean([p[0] for p in pts])), 2),
                  round(float(np.mean([p[1] for p in pts])), 2)]
            for key, pts in group_points.items() if pts
        }

        frames.append({
            'frame_id': int(frame_id),
            'time': str(frame_df['time'].iloc[0]) if 'time' in frame_df else None,
            'event': (frame_df['event'].dropna().iloc[0]
                      if 'event' in frame_df.columns and frame_df['event'].notna().any()
                      else None),
            'players': players,
            'control_areas': control_areas,
            'averages': averages,
        })

    player_routes = build_player_route_profiles(play_df, direction_mult, ball_x_snap, ball_y_snap)

    return {'meta': meta, 'frames': frames, 'player_routes': player_routes}


def normalize_frame_id(df):
    """frame_id may arrive as a composite key like '2025090400-648-160'
    (game_id-gsis_play_id-frame_number) rather than a plain int. Extract
    just the trailing frame number so downstream grouping/sorting works.
    Runs regardless of the column's reported dtype, since a mixed-type
    column can still come back as int64/float64 if pandas coerces it."""
    def parse(val):
        s = str(val)
        if '-' in s:
            return int(s.split('-')[-1])
        return int(float(s))

    print("frame_id sample before normalize:", df['frame_id'].iloc[:5].tolist(),
          "dtype:", df['frame_id'].dtype)
    df['frame_id'] = df['frame_id'].apply(parse)
    print("frame_id sample after normalize:", df['frame_id'].iloc[:5].tolist())
    return df


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


if __name__ == '__main__':
    csv_path, game_id, gsis_play_id, out_path = sys.argv[1:5]
    df = pd.read_csv(csv_path)
    df = normalize_frame_id(df)
    result = build_play_json(df, int(game_id), int(gsis_play_id))
    with open(out_path, 'w') as f:
        json.dump(result, f, default=_json_default)
    print(f"Wrote {len(result['frames'])} frames to {out_path}")