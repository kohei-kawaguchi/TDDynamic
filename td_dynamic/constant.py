from bidict import bidict

EulerGamma = 0.57721566490153286060651209008240243104215933593992


FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters
STOP_THRESHOLD = 0.1  # unit: m/s

HOME_AWAY_MAP = {
    0: "BALL",
    1: "HOME",
    2: "AWAY",
}

PLAYER_ROLE_MAP = bidict(
    {
        1: "GK",
        2: "DF",
        3: "MF",
        4: "FW",
    }
)

INPUT_EVENT_COLUMNS = [
    "game_id",
    "frame_id",
    "absolute_time",
    "match_status_id",
    "home_away",
    "event_x",
    "event_y",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "jersey_number",
    "player_role_id",
    "event_id",
    "event_name",
    "ball_x",
    "ball_y",
    "attack_history_num",
    "attack_direction",
    "series_num",
    "ball_touch",
    "success",
    "history_num",
    "attack_start_history_num",
    "attack_end_history_num",
    "is_goal",
    "is_shot",
    "is_pass",
    "is_cross",
    "is_through_pass",
    "is_dribble",
]

INPUT_TRACKING_COLUMNS = [
    "game_id",
    "frame_id",
    "home_away",
    "jersey_number",
    "x",
    "y",
]

INPUT_PLAYER_COLUMNS = [
    "home_away",
    "team_id",
    "player_id",
    "player_name",
    "player_role",
    "jersey_number",
    "starting_member",
    "on_pitch",
]