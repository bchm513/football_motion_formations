import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator
import math

################################################
# Load and Filter Play by Play Data
################################################

all_plays = pd.read_csv('player_play_routes.csv')
all_plays = all_plays[all_plays['play_type'] == 'play_type_pass']
unique_plays = all_plays[['game_id', 'gsis_play_id']].drop_duplicates()
unique_plays

################################################
# Pair Play by Play Data with Tracking Data
################################################

# Set authorization token - replace with your own token
auth_token = "NGS AKIAY5YRP3ZL7RRDSWVW:R6OWqOWZBLLJsM6m50o1dQuTbvU="

# Pull the plays data (replace with your own path) - this should include position info, nfl ids, game ids, and play ids at a minimum. We will use the game and play ids to pull tracking data
play_info = all_plays # -> Use a SQL query to pull play IDs

# Function to pull tracking data for a given player and play
def pull_tracking_data(game_id, play_id):

    # Testing - Use these values if you want to run anything within the function directly
    # game_id = 2025090400
    # play_id = 71

    url = f"https://api.ngs.nfl.com/tracking/game/play?gameId={game_id}&gsisPlayId={play_id}"
    
    headers = {
        "Authorization": auth_token
    }
    
    response = requests.get(url, headers=headers)
    res_json = response.json()
    # print(res_json)

    # Extract the home and away tracking data
    home_tracking = pd.json_normalize(res_json['homeTrackingData']).assign(game_id=game_id, gsis_play_id=play_id)
    away_tracking = pd.json_normalize(res_json['awayTrackingData']).assign(game_id=game_id, gsis_play_id=play_id)
    # print(away_tracking)

    # Merge the tracking data
    tracking = pd.concat([home_tracking, away_tracking], ignore_index=True)

    # Unnest the playerTrackingData column
    if 'playerTrackingData' in tracking.columns:
        player_tracking_list = []
        for idx, row in tracking.iterrows():
            if isinstance(row['playerTrackingData'], list):
                player_df = pd.json_normalize(row['playerTrackingData'])
                # Add all other columns from the parent row
                for col in tracking.columns:
                    if col != 'playerTrackingData':
                        player_df[col] = row[col]
                player_tracking_list.append(player_df)
        
        if player_tracking_list:
            tracking = pd.concat(player_tracking_list, ignore_index=True)

    # print(tracking)
    # Filter to what we need - expand or contact this list as necessary
    tracking = tracking[['time', 'a', 'dir', 'dis','s', 'event', 'x', 'y', 'nflId', 'displayName', 'position', 'positionGroup', 'game_id', 'gsis_play_id']]

    numeric_cols = ['a', 'dir', 'dis', 's', 'x', 'y', 'game_id', 'gsis_play_id', 'nflId']  # List your numeric columns
    for col in numeric_cols:
        tracking[col] = pd.to_numeric(tracking[col], errors = 'coerce')  # Use 'coerce' to turn invalids to NaN

    # Merge with play_info to get player positions (ngs_position) and other details
    tracking_df_merged = tracking.merge(
        play_info,
        left_on=['game_id', 'gsis_play_id', 'nflId'],
        right_on=['game_id', 'gsis_play_id', 'nfl_id'],
        how='left'
    )

    # Convert time to datetime
    tracking_df_merged['time'] = pd.to_datetime(tracking_df_merged['time'], format="%Y-%m-%dT%H:%M:%S.%f", utc=True, errors='coerce')

    # Create frame_id (group by time) - this is helpful for frame identification
    tracking_df_merged['frame_id'] = (
        tracking_df_merged.groupby('time')
        .ngroup()
        .astype(str)
    )
    tracking_df_merged['frame_id'] = (
        tracking_df_merged['game_id'].astype(str) + "-" +
        tracking_df_merged['gsis_play_id'].astype(str) + "-" +
        tracking_df_merged['frame_id'].astype(str).str.zfill(3)
    )

    # ***OPTIONAL: In this section, you can cut for specific frames (i.e. post ball_snap) using the events column***

    return tracking_df_merged

    # tracking_df_merged will be all your tracking data for a particular play. to get multiple plays and games, you would loop through games ids as well and then nest the play id loop within it (see below)

tracking_data = pd.DataFrame()

for _, row in unique_plays.iterrows():
    if row['game_id'] == 2025121408:
        # print('ere')
        tracking_data_pulled = pull_tracking_data(game_id=row['game_id'], play_id=row['gsis_play_id'])
        tracking_data = pd.concat([tracking_data, tracking_data_pulled], ignore_index=True)

tracking_data = tracking_data[(tracking_data['offense_play'] == True) 
                                & (tracking_data['was_running_route'] == True)
                                & (tracking_data['route'] != 'GO')
                                # & (tracking_data['pass_blocking'] != 'True')
                              ]

################################################
# Functions to Identify Route Breaks
################################################

def shift_frame(frame_key: str, offset: int = 2) -> str:
    # Split from the right on the last hyphen only
    prefix, frame_str = frame_key.rsplit('-', 1)
    
    # Subtract offset and keep original digit padding length
    # print('frame_str', frame_str)
    new_frame = int(frame_str) - offset
    # print('new_frame', new_frame)
    padded_frame = str(new_frame).zfill(len(frame_str))
    
    return f"{prefix}-{padded_frame}"

def detect_route_break(df_player, min_angle_change=5, window=5, group_gap=10, 
                       enable_late_window_fallback=True, fallback_lookback=5, fallback_min_rise=.2):
    df_player = df_player.sort_values('frame_id').reset_index(drop=True)

    # Handle angle wrapping (0 to 360 deg)
    dir_rad = np.radians(df_player['dir'])
    sin_mean = np.sin(dir_rad).rolling(window, center=True, min_periods=1).mean()
    cos_mean = np.cos(dir_rad).rolling(window, center=True, min_periods=1).mean()
    smoothed_dir = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360

    # Calculate absolute frame-to-frame change in heading
    angle_diff = np.abs(np.diff(smoothed_dir, prepend=smoothed_dir.iloc[0]))
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)

    df_player['angle_change_rate'] = angle_diff

    # Find peaks in directional change rate
    peaks, properties = find_peaks(
        df_player['angle_change_rate'],
        height=1,
        distance=1
    )

    # Filter candidate peaks by total angle shift in surrounding window
    break_indices_candidates = [] 

    direction_mult = -1 if df_player['play_direction'].iloc[0] == 'left' else 1
    los_x_raw = df_player['x_ball_at_snap'].iloc[0]
    los_x = los_x_raw + 1 if df_player['play_direction'].iloc[0] == 'left' else los_x_raw - 1

    def _passes_los(x_val):
        if los_x is None:
            return True
        downfield_dist = direction_mult * (x_val - los_x)
        # print('downfield_dist', downfield_dist, 'x_val', x_val, 'los_x', los_x, 'direction_mult', direction_mult)
        return round(downfield_dist) > 1

    for peak in peaks:
        start_idx = max(0, peak - 1)
        end_idx = min(len(df_player) - 1, peak + 1)
        total_turn = np.sum(df_player['angle_change_rate'].iloc[start_idx:end_idx + 1])
        peak_frame = df_player['frame_id'].iloc[peak]
        peak_x = df_player['x'].iloc[peak]

        if not _passes_los(peak_x):
            if _passes_los(df_player['x'].iloc[-1]):
                break_indices_candidates.append({'peak': peak, 'frame': peak_frame, 'total_turn': total_turn})
            else:
                continue

        if total_turn >= min_angle_change:
            break_indices_candidates.append({'peak': peak, 'frame': peak_frame, 'total_turn': total_turn})

    break_frames = []
    break_indices = []
    break_angles = []

    current_group = []
    for cand in break_indices_candidates:
        # print(cand)
        if not current_group or (cand['peak'] - current_group[-1]['peak']) <= group_gap:
            current_group.append(cand)
        else:
            best = max(current_group, key=lambda c: c['total_turn'])
            break_frames.append((shift_frame(best['frame'], offset=2), best['peak'], best['total_turn']))
            break_indices.append(best['peak'])
            break_angles.append(df_player['angle_change_rate'].iloc[best['peak']])
            current_group = [cand]

    # don't forget the last group
    if current_group:
        best = max(current_group, key=lambda c: c['total_turn'])
        break_frames.append((shift_frame(best['frame'], offset=2), best['peak'], best['total_turn']))
        break_indices.append(best['peak'])
        break_angles.append(df_player['angle_change_rate'].iloc[best['peak']])

    # if there is no break detected, check for a late window fallback 
    # (they are speeding up as the window ends and do not have enough frames to come back down and form a peak)
    if enable_late_window_fallback and not break_frames:
        lookback = fallback_lookback if fallback_lookback is not None else window
        min_rise = fallback_min_rise if fallback_min_rise is not None else min_angle_change
        # print('nada', df_player['displayName'].iloc[0])
 
        n = len(df_player)
        tail_n = min(lookback, n)
        # print('tail_n', tail_n, 'n', n, 'lookback', lookback)
 
        if tail_n >= 3:
            raw_dir = df_player['dir'].values
            raw_diff = np.abs(np.diff(raw_dir, prepend=raw_dir[0]))
            raw_diff = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)

            trailing_rate = pd.Series(raw_diff).rolling(lookback, min_periods=1, center=False).mean().values
 
            tail_start = n - tail_n
            tail_rates = trailing_rate[tail_start:]
            tail_x = np.arange(tail_n)
 
            slope = np.polyfit(tail_x, tail_rates, 1)[0]
            recent_rise = tail_rates[-1] - tail_rates[0]
            # print(slope, recent_rise, min_rise)
 
            is_dramatically_rising = slope > 0 and recent_rise >= min_rise
            median_rate = np.median(df_player['angle_change_rate'])
            end_rate = df_player['angle_change_rate'].iloc[-1]
            print(end_rate, median_rate, is_dramatically_rising, df_player['displayName'].iloc[0], df_player['gsis_play_id'].iloc[0])
 
            if is_dramatically_rising:
                fallback_idx = max(0, n - 1 - 2)  # 2 frames before the end of the window
                fallback_x = df_player['x'].iloc[fallback_idx]
 
                if _passes_los(fallback_x):
                    fallback_frame = df_player['frame_id'].iloc[fallback_idx]
                    start_idx = max(0, fallback_idx - 1)
                    end_idx = min(n - 1, fallback_idx + 1)
                    fallback_turn = float(np.sum(df_player['angle_change_rate'].iloc[start_idx:end_idx + 1]))
 
                    break_frames.append((shift_frame(fallback_frame, offset=2), fallback_idx, fallback_turn))
                    break_indices.append(fallback_idx)
                    break_angles.append(df_player['angle_change_rate'].iloc[fallback_idx])

            if not break_frames:
                print(f"No route break detected for player {df_player['displayName'].iloc[0]} in play {df_player['gsis_play_id'].iloc[0]}.")
                median_rate = np.nanmedian(df_player['angle_change_rate'])
                end_rate = df_player['angle_change_rate'].iloc[-1]
                if end_rate > 2 and median_rate > 2:
                    print(f"Player {df_player['displayName'].iloc[0]} in play {df_player['gsis_play_id'].iloc[0]} has end rate {end_rate} above median rate {median_rate}.")

    return df_player, peaks, break_frames, break_indices, break_angles

################################################
# Loop through Data and Determine Route Breaks
################################################

unique_player_plays = tracking_data[['game_id', 'gsis_play_id', 'nfl_id']].drop_duplicates()

break_frames_final = []

for _, row in unique_player_plays.iterrows():
    result_rows = []

    player_df = tracking_data[
        (tracking_data['game_id'] == row['game_id']) &
        (tracking_data['gsis_play_id'] == row['gsis_play_id']) &
        (tracking_data['nfl_id'] == row['nfl_id'])
    ].sort_values('frame_id')

    snap_frames = player_df.loc[player_df['event'] == 'ball_snap', 'time']
    end_events = ['pass_outcome_caught', 'pass_arrived', 'out_of_bounds', 'pass_outcome_incomplete',
                    'dropped_pass', 'pass_tipped', 'qb_spike']

    submit_frames = player_df.loc[player_df['event'].isin(end_events), 'time'].sort_values()

    if submit_frames.empty:
        submit_frames = player_df.loc[player_df['event'] == 'play_submit', 'time']

    # skip if this player/play doesn't have both events
    if snap_frames.empty or submit_frames.empty:
        continue

    snap_frame = snap_frames.iloc[0]
    submit_frame = submit_frames.iloc[0]

    mask = (player_df['time'] >= snap_frame) & (player_df['time'] <= submit_frame)
    result_rows.append(player_df.loc[mask])
    result_df = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()

    df_player, peaks, break_frames, break_indices, break_angles = detect_route_break(result_df)
    # print(row['nfl_id'], df_player[['x_ball_at_snap', 'y_ball_at_snap', 'play_direction']].iloc[0].to_dict())
    
    # Function for plotting routes, not required for this report
    # plot_route_with_breaks(df_player, peaks, break_indices)
    
    if break_frames:
        # print(break_angles, 'break_angles', break_frames, 'break_frames')
        mask = (
            (tracking_data['game_id'] == row['game_id']) &
            (tracking_data['gsis_play_id'] == row['gsis_play_id']) &
            (tracking_data['nfl_id'] == row['nfl_id'])
        )
        tracking_data.loc[mask, 'route_break_frame'] = break_frames[0][0]

    # print(break_frames[0], 'asdfasdf')
    if len(break_frames) > 1: # if there are multiple break frames, take the highest peak one (the first is usually a false positive)
        final_break_frame = max(break_frames, key=lambda b: b[2])
    elif len(break_frames) == 1: # if there is only one break frame, take that one, convert from list
        final_break_frame = break_frames[0]
    else:
        final_break_frame = break_frames # else get the empty item

    break_frames_final.append((row['game_id'], row['gsis_play_id'], row['nfl_id'],
                               result_df['displayName'].iloc[0], final_break_frame, result_df['play_direction'].iloc[0]))
    # print(result_df['displayName'].iloc[0], break_frames)
    # print(len(break_frames))
    print((row['game_id'], row['gsis_play_id'], row['nfl_id'], result_df['displayName'].iloc[0], final_break_frame))


################################################
# Loop through Results and Return Route Breaks
################################################

records = []

for game_id, gsis_play_id, nfl_id, display_name, final_break_frame, play_direction in break_frames_final:
    player_df = tracking_data[
        (tracking_data['game_id'] == game_id) &
        (tracking_data['gsis_play_id'] == gsis_play_id) &
        (tracking_data['nfl_id'] == nfl_id)
    ].sort_values('frame_id')

    if player_df.empty:
        continue

    # final_break_frame is either a tuple (frame_id, peak_idx, total_turn)
    # or an empty list (no break found)
    if final_break_frame and isinstance(final_break_frame, tuple):
        direction_mult = 1 if play_direction == 'right' else -1
        break_frame_id = final_break_frame[0]
        route_break_depth = round(
            direction_mult * (
                player_df.loc[player_df['frame_id'] == break_frame_id, 'x'].values[0]
                - player_df['x_ball_at_snap'].values[0]
            )
        )
    else:
        break_frame_id = None
        route_break_depth = None

    records.append({
        'game_id': game_id,
        'gsis_play_id': gsis_play_id,
        'nfl_id': nfl_id,
        'display_name': display_name,
        'route_break_frame': break_frame_id,
        'route_break_depth': route_break_depth
    })

tracking_data_play = pd.DataFrame(records)
tracking_data_play.head(50)