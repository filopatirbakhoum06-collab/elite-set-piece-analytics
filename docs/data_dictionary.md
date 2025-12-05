# Data Dictionary

## Overview

This document describes the data structures and fields used throughout
the Elite Set-Piece Analytics platform.

---

## Event Data

### Standard Event Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `event_id` | string | Unique event identifier | "ev_123456" |
| `match_id` | string | Match identifier | "match_789" |
| `event_type` | string | Type of event | "Corner", "Pass" |
| `player_id` | string | Player identifier | "player_42" |
| `player_name` | string | Player's name | "Lionel Messi" |
| `team_id` | string | Team identifier | "team_01" |
| `team_name` | string | Team name | "Argentina" |
| `x` | float | X coordinate (meters) | 45.5 |
| `y` | float | Y coordinate (meters) | 32.0 |
| `end_x` | float | End X coordinate | 88.3 |
| `end_y` | float | End Y coordinate | 40.1 |
| `timestamp` | float | Match time (seconds) | 1234.5 |
| `period` | int | Match period (1 or 2) | 2 |
| `minute` | int | Match minute | 67 |
| `second` | int | Second within minute | 23 |

### Event Types

| Event Type | Description |
|------------|-------------|
| Pass | Ball passed between players |
| Shot | Attempt on goal |
| Corner | Corner kick |
| Free Kick | Free kick |
| Throw-in | Throw-in |
| Penalty | Penalty kick |
| Dribble | Player dribbles ball |
| Foul | Foul committed |
| Goal | Goal scored |

---

## Set-Piece Data

### Set-Piece Event Schema

Extends Standard Event Schema with:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `set_piece_type` | string | Type of set-piece | "corner" |
| `delivery_type` | string | How ball is delivered | "inswinging" |
| `first_receiver` | string | Player receiving ball | "player_10" |
| `zone` | string | Pitch zone | "attacking_center" |
| `outcome` | string | Result of set-piece | "shot" |

### Corner Kick Fields

| Field | Type | Description |
|-------|------|-------------|
| `corner_side` | string | "left" or "right" |
| `corner_type` | string | "standard", "short", "near_post", "far_post" |
| `in_swinging` | bool | Whether delivery curves toward goal |
| `n_attackers_in_box` | int | Attackers in penalty area |
| `n_defenders_in_box` | int | Defenders in penalty area |

### Free Kick Fields

| Field | Type | Description |
|-------|------|-------------|
| `distance_to_goal` | float | Distance from goal (meters) |
| `free_kick_type` | string | "direct" or "indirect" |
| `wall_players` | int | Number in defensive wall |
| `shooting_angle` | float | Angle to goal (degrees) |

---

## Tracking Data

### Frame Schema

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Frame number |
| `timestamp` | float | Time in seconds |
| `ball_x` | float | Ball X position |
| `ball_y` | float | Ball Y position |
| `player_1_x` | float | Player 1 X position |
| `player_1_y` | float | Player 1 Y position |
| ... | ... | (up to 22 players) |

---

## Feature Data

### Spatial Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `distance_to_goal` | float | Distance to attacking goal | 0-120 m |
| `angle_to_goal` | float | Angle to goal center | 0-90° |
| `visible_goal_angle` | float | Visible goal width angle | 0-180° |
| `distance_to_ball` | float | Distance from ball | 0-150 m |
| `zone` | string | Pitch zone classification | 9 zones |
| `in_penalty_area` | bool | Inside penalty area | True/False |
| `density_5m` | int | Players within 5 meters | 0-22 |
| `density_10m` | int | Players within 10 meters | 0-22 |
| `nearest_teammate_dist` | float | Distance to nearest teammate | 0-100 m |
| `nearest_opponent_dist` | float | Distance to nearest opponent | 0-100 m |

### Temporal Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `time_until_end` | float | Seconds until half/match end | 0-2700 s |
| `time_pressure` | float | Time pressure factor | 0-1 |
| `event_duration` | float | Duration since last event | 0-inf s |
| `sequence_position` | int | Position in possession | 1-inf |
| `tempo` | float | Events per minute (rolling) | 0-100 |

### Physical Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `speed` | float | Current speed | 0-12 m/s |
| `acceleration` | float | Current acceleration | -10 to 10 m/s² |
| `direction` | float | Movement direction | 0-360° |
| `direction_change` | float | Rate of direction change | 0-180°/frame |
| `distance_covered` | float | Total distance | 0-inf m |

---

## Model Outputs

### Receiver Prediction Output

| Field | Type | Description |
|-------|------|-------------|
| `predicted_receiver` | string | Most likely receiver |
| `probabilities` | array | Probability per player |
| `top_k_receivers` | array | Top K receivers with probs |
| `confidence` | float | Model confidence (0-1) |

### Outcome Prediction Output

| Field | Type | Description |
|-------|------|-------------|
| `goal_probability` | float | xG-like probability (0-1) |
| `shot_probability` | float | Probability of shot (0-1) |
| `outcome_class` | string | Predicted outcome category |

---

## Pitch Coordinates

### Coordinate System

- Origin: Bottom-left corner (defending goal line)
- X-axis: Along the length of the pitch (0 to 105)
- Y-axis: Along the width of the pitch (0 to 68)
- Units: Meters

### Key Positions

| Location | X | Y |
|----------|---|---|
| Center | 52.5 | 34 |
| Left Goal | 0 | 34 |
| Right Goal | 105 | 34 |
| Left Penalty Spot | 11 | 34 |
| Right Penalty Spot | 94 | 34 |
| Left Corner (bottom) | 0 | 0 |
| Left Corner (top) | 0 | 68 |
| Right Corner (bottom) | 105 | 0 |
| Right Corner (top) | 105 | 68 |

---

## Data Quality Notes

1. **Missing Values**: Represented as `NaN` or `None`
2. **Coordinate Normalization**: Different sources use different scales
3. **Time Synchronization**: Tracking and event data may need alignment
4. **Player Identification**: IDs vary by data source
