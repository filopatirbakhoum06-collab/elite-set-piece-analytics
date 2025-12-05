"""
Data loading utilities for Elite Set-Piece Analytics.
أدوات تحميل البيانات

This module provides loaders for various football data sources including:
- Wyscout event data
- StatsBomb open data
- Metrica tracking data
- Unified data format

Example usage:
    >>> from src.data.loaders import WyscoutLoader, StatsBombLoader
    >>> 
    >>> # Load Wyscout data
    >>> wyscout = WyscoutLoader()
    >>> events = wyscout.load_events(match_id=123456)
    >>> 
    >>> # Load StatsBomb data
    >>> statsbomb = StatsBombLoader()
    >>> events = statsbomb.load_match_events(match_id=789)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    الفئة الأساسية المجردة لجميع محملات البيانات
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory. If None, uses default from config.
        """
        from ..utils.config import Config
        self.data_dir = data_dir or Config.RAW_DATA_DIR
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def load_events(self, match_id: Union[int, str]) -> pd.DataFrame:
        """Load event data for a specific match."""
        pass
    
    @abstractmethod
    def load_matches(self, competition_id: Optional[int] = None) -> pd.DataFrame:
        """Load match metadata."""
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if df.empty:
            self.logger.warning("Loaded DataFrame is empty")
            return False
        return True


class WyscoutLoader(BaseDataLoader):
    """
    Data loader for Wyscout event data.
    محمل بيانات Wyscout
    
    Wyscout provides detailed event data including:
    - Passes, shots, tackles, dribbles
    - Player and team information
    - Coordinates and timestamps
    
    Example:
        >>> loader = WyscoutLoader()
        >>> events = loader.load_events(match_id=2499719)
        >>> print(events.columns)
    """
    
    # Event type mappings
    EVENT_TYPES = {
        1: "Duel",
        2: "Foul",
        3: "Free Kick",
        4: "Goal Keeper",
        5: "Interruption",
        6: "Offside",
        7: "Others on the ball",
        8: "Pass",
        9: "Shot",
        10: "Save Attempt",
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the Wyscout data loader."""
        super().__init__(data_dir)
        self.wyscout_dir = self.data_dir / "wyscout"
    
    def load_events(self, match_id: Union[int, str]) -> pd.DataFrame:
        """
        Load event data for a specific match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            DataFrame containing match events
        """
        events_file = self.wyscout_dir / "events" / f"events_{match_id}.json"
        
        if not events_file.exists():
            self.logger.warning(f"Events file not found: {events_file}")
            return pd.DataFrame()
        
        with open(events_file, "r") as f:
            events_data = json.load(f)
        
        df = pd.DataFrame(events_data)
        df = self._process_events(df)
        
        self.validate_data(df)
        return df
    
    def load_matches(
        self, 
        competition_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load match metadata.
        
        Args:
            competition_id: Optional competition filter
            
        Returns:
            DataFrame containing match information
        """
        matches_file = self.wyscout_dir / "matches" / "matches.json"
        
        if not matches_file.exists():
            self.logger.warning(f"Matches file not found: {matches_file}")
            return pd.DataFrame()
        
        with open(matches_file, "r") as f:
            matches_data = json.load(f)
        
        df = pd.DataFrame(matches_data)
        
        if competition_id is not None:
            df = df[df["competitionId"] == competition_id]
        
        return df
    
    def load_players(self) -> pd.DataFrame:
        """Load player data."""
        players_file = self.wyscout_dir / "players.json"
        
        if not players_file.exists():
            self.logger.warning(f"Players file not found: {players_file}")
            return pd.DataFrame()
        
        with open(players_file, "r") as f:
            players_data = json.load(f)
        
        return pd.DataFrame(players_data)
    
    def load_teams(self) -> pd.DataFrame:
        """Load team data."""
        teams_file = self.wyscout_dir / "teams.json"
        
        if not teams_file.exists():
            self.logger.warning(f"Teams file not found: {teams_file}")
            return pd.DataFrame()
        
        with open(teams_file, "r") as f:
            teams_data = json.load(f)
        
        return pd.DataFrame(teams_data)
    
    def _process_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw Wyscout events.
        
        Args:
            df: Raw events DataFrame
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        # Add event type names
        if "eventId" in df.columns:
            df["event_type"] = df["eventId"].map(self.EVENT_TYPES)
        
        # Extract positions if available
        if "positions" in df.columns:
            df["start_x"] = df["positions"].apply(
                lambda x: x[0]["x"] if len(x) > 0 else None
            )
            df["start_y"] = df["positions"].apply(
                lambda x: x[0]["y"] if len(x) > 0 else None
            )
            df["end_x"] = df["positions"].apply(
                lambda x: x[1]["x"] if len(x) > 1 else None
            )
            df["end_y"] = df["positions"].apply(
                lambda x: x[1]["y"] if len(x) > 1 else None
            )
        
        return df


class StatsBombLoader(BaseDataLoader):
    """
    Data loader for StatsBomb open data.
    محمل بيانات StatsBomb
    
    StatsBomb provides high-quality event data including:
    - Detailed event attributes
    - 360 data (player positions)
    - Expected goals (xG)
    
    Example:
        >>> loader = StatsBombLoader()
        >>> events = loader.load_match_events(match_id=3788741)
        >>> print(events.shape)
    """
    
    def __init__(self, data_dir: Optional[Path] = None, use_api: bool = True):
        """
        Initialize the StatsBomb data loader.
        
        Args:
            data_dir: Path to data directory
            use_api: Whether to use statsbombpy API (requires internet)
        """
        super().__init__(data_dir)
        self.use_api = use_api
        self.statsbomb_dir = self.data_dir / "statsbomb"
    
    def load_events(self, match_id: Union[int, str]) -> pd.DataFrame:
        """Load event data for a specific match."""
        return self.load_match_events(match_id)
    
    def load_match_events(self, match_id: Union[int, str]) -> pd.DataFrame:
        """
        Load detailed event data for a match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            DataFrame containing match events
        """
        if self.use_api:
            try:
                from statsbombpy import sb
                events = sb.events(match_id=int(match_id))
                return events
            except ImportError:
                self.logger.warning("statsbombpy not installed, using local files")
            except Exception as e:
                self.logger.warning(f"API call failed: {e}, using local files")
        
        # Fallback to local files
        events_file = self.statsbomb_dir / "events" / f"{match_id}.json"
        
        if not events_file.exists():
            self.logger.warning(f"Events file not found: {events_file}")
            return pd.DataFrame()
        
        with open(events_file, "r") as f:
            events_data = json.load(f)
        
        return pd.DataFrame(events_data)
    
    def load_matches(
        self, 
        competition_id: Optional[int] = None,
        season_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load match metadata.
        
        Args:
            competition_id: Competition identifier
            season_id: Season identifier
            
        Returns:
            DataFrame containing match information
        """
        if self.use_api and competition_id is not None and season_id is not None:
            try:
                from statsbombpy import sb
                matches = sb.matches(
                    competition_id=competition_id,
                    season_id=season_id
                )
                return matches
            except ImportError:
                self.logger.warning("statsbombpy not installed")
            except Exception as e:
                self.logger.warning(f"API call failed: {e}")
        
        return pd.DataFrame()
    
    def load_competitions(self) -> pd.DataFrame:
        """Load available competitions."""
        if self.use_api:
            try:
                from statsbombpy import sb
                return sb.competitions()
            except ImportError:
                self.logger.warning("statsbombpy not installed")
            except Exception as e:
                self.logger.warning(f"API call failed: {e}")
        
        return pd.DataFrame()
    
    def load_lineups(self, match_id: Union[int, str]) -> Dict[str, pd.DataFrame]:
        """
        Load lineup data for a match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            Dictionary with team names as keys and lineup DataFrames as values
        """
        if self.use_api:
            try:
                from statsbombpy import sb
                return sb.lineups(match_id=int(match_id))
            except ImportError:
                self.logger.warning("statsbombpy not installed")
            except Exception as e:
                self.logger.warning(f"API call failed: {e}")
        
        return {}


class MetricaLoader(BaseDataLoader):
    """
    Data loader for Metrica Sports tracking data.
    محمل بيانات Metrica للتتبع
    
    Metrica provides:
    - Player tracking data (x, y coordinates)
    - Ball tracking data
    - Event data synchronized with tracking
    
    Example:
        >>> loader = MetricaLoader()
        >>> tracking = loader.load_tracking_data(match_id=1)
        >>> print(tracking.columns)
    """
    
    SAMPLE_MATCHES = {
        1: "Sample_Game_1",
        2: "Sample_Game_2",
        3: "Sample_Game_3",
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the Metrica data loader."""
        super().__init__(data_dir)
        self.metrica_dir = self.data_dir / "metrica"
    
    def load_events(self, match_id: Union[int, str]) -> pd.DataFrame:
        """Load event data for a specific match."""
        match_name = self.SAMPLE_MATCHES.get(int(match_id), str(match_id))
        events_file = self.metrica_dir / match_name / "events.csv"
        
        if not events_file.exists():
            self.logger.warning(f"Events file not found: {events_file}")
            return pd.DataFrame()
        
        return pd.read_csv(events_file)
    
    def load_matches(
        self, 
        competition_id: Optional[int] = None
    ) -> pd.DataFrame:
        """List available matches."""
        matches = []
        for match_id, match_name in self.SAMPLE_MATCHES.items():
            matches.append({
                "match_id": match_id,
                "match_name": match_name,
            })
        return pd.DataFrame(matches)
    
    def load_tracking_data(
        self, 
        match_id: Union[int, str],
        team: str = "home"
    ) -> pd.DataFrame:
        """
        Load tracking data for a match.
        
        Args:
            match_id: Match identifier
            team: "home" or "away"
            
        Returns:
            DataFrame containing tracking data
        """
        match_name = self.SAMPLE_MATCHES.get(int(match_id), str(match_id))
        tracking_file = self.metrica_dir / match_name / f"tracking_{team}.csv"
        
        if not tracking_file.exists():
            self.logger.warning(f"Tracking file not found: {tracking_file}")
            return pd.DataFrame()
        
        return pd.read_csv(tracking_file)
    
    def load_ball_tracking(self, match_id: Union[int, str]) -> pd.DataFrame:
        """Load ball tracking data."""
        match_name = self.SAMPLE_MATCHES.get(int(match_id), str(match_id))
        ball_file = self.metrica_dir / match_name / "tracking_ball.csv"
        
        if not ball_file.exists():
            # Ball data might be in home tracking file
            return pd.DataFrame()
        
        return pd.read_csv(ball_file)


class UnifiedDataLoader:
    """
    Unified data loader that provides consistent interface across data sources.
    محمل البيانات الموحد
    
    This loader normalizes data from different sources into a consistent format,
    making it easy to work with multiple data providers.
    
    Example:
        >>> loader = UnifiedDataLoader()
        >>> events = loader.load_events(source="statsbomb", match_id=3788741)
        >>> set_pieces = loader.get_set_pieces(events)
    """
    
    # Standard column names
    STANDARD_COLUMNS = {
        "match_id": "match_id",
        "event_id": "event_id",
        "event_type": "event_type",
        "player_id": "player_id",
        "player_name": "player_name",
        "team_id": "team_id",
        "team_name": "team_name",
        "x": "x",
        "y": "y",
        "end_x": "end_x",
        "end_y": "end_y",
        "timestamp": "timestamp",
        "period": "period",
        "minute": "minute",
        "second": "second",
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the unified data loader.
        
        Args:
            data_dir: Base data directory
        """
        from ..utils.config import Config
        self.data_dir = data_dir or Config.RAW_DATA_DIR
        
        # Initialize individual loaders
        self.loaders = {
            "wyscout": WyscoutLoader(self.data_dir),
            "statsbomb": StatsBombLoader(self.data_dir),
            "metrica": MetricaLoader(self.data_dir),
        }
    
    def load_events(
        self, 
        source: str, 
        match_id: Union[int, str],
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Load events from a specific source.
        
        Args:
            source: Data source name ("wyscout", "statsbomb", "metrica")
            match_id: Match identifier
            normalize: Whether to normalize column names
            
        Returns:
            DataFrame containing events
        """
        if source not in self.loaders:
            raise ValueError(f"Unknown source: {source}. Available: {list(self.loaders.keys())}")
        
        events = self.loaders[source].load_events(match_id)
        
        if normalize and not events.empty:
            events = self._normalize_events(events, source)
        
        return events
    
    def _normalize_events(
        self, 
        df: pd.DataFrame, 
        source: str
    ) -> pd.DataFrame:
        """
        Normalize events to standard format.
        
        Args:
            df: Raw events DataFrame
            source: Data source name
            
        Returns:
            Normalized DataFrame
        """
        if source == "wyscout":
            return self._normalize_wyscout(df)
        elif source == "statsbomb":
            return self._normalize_statsbomb(df)
        elif source == "metrica":
            return self._normalize_metrica(df)
        else:
            return df
    
    def _normalize_wyscout(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Wyscout data to standard format."""
        column_mapping = {
            "matchId": "match_id",
            "id": "event_id",
            "eventId": "event_type_id",
            "event_type": "event_type",
            "playerId": "player_id",
            "teamId": "team_id",
            "start_x": "x",
            "start_y": "y",
            "end_x": "end_x",
            "end_y": "end_y",
            "eventSec": "timestamp",
            "matchPeriod": "period",
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Convert Wyscout coordinates (0-100) to meters
        from ..utils.config import Config
        if "x" in df.columns:
            df["x"] = df["x"] * Config.PITCH_LENGTH / 100
        if "y" in df.columns:
            df["y"] = df["y"] * Config.PITCH_WIDTH / 100
        if "end_x" in df.columns:
            df["end_x"] = df["end_x"] * Config.PITCH_LENGTH / 100
        if "end_y" in df.columns:
            df["end_y"] = df["end_y"] * Config.PITCH_WIDTH / 100
        
        return df
    
    def _normalize_statsbomb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize StatsBomb data to standard format."""
        column_mapping = {
            "match_id": "match_id",
            "id": "event_id",
            "type": "event_type",
            "player": "player_name",
            "player_id": "player_id",
            "team": "team_name",
            "team_id": "team_id",
            "location": "location",
            "timestamp": "timestamp",
            "period": "period",
            "minute": "minute",
            "second": "second",
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Extract x, y from location array
        if "location" in df.columns:
            df["x"] = df["location"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
            )
            df["y"] = df["location"].apply(
                lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None
            )
        
        return df
    
    def _normalize_metrica(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Metrica data to standard format."""
        column_mapping = {
            "Match_id": "match_id",
            "Event_id": "event_id",
            "Type": "event_type",
            "From": "player_name",
            "Team": "team_name",
            "Start X": "x",
            "Start Y": "y",
            "End X": "end_x",
            "End Y": "end_y",
            "Start Time [s]": "timestamp",
            "Period": "period",
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        return df
    
    def get_set_pieces(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract set-piece events from an events DataFrame.
        
        Args:
            events: Events DataFrame
            
        Returns:
            DataFrame containing only set-piece events
        """
        set_piece_types = [
            "Corner",
            "Free Kick",
            "Throw-in",
            "Penalty",
            "Goal Kick",
            "corner",
            "free_kick",
            "throw_in",
            "penalty",
            "goal_kick",
        ]
        
        if "event_type" not in events.columns:
            return pd.DataFrame()
        
        # Filter set pieces
        mask = events["event_type"].astype(str).str.lower().str.contains(
            "|".join([sp.lower() for sp in set_piece_types]), 
            na=False
        )
        
        return events[mask].copy()
