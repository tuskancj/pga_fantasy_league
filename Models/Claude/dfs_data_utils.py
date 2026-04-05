"""
DFS Data Utilities
Reusable functions for PGA DFS data cleaning, imputation, and feature engineering.
Author: Claude
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class DFSDataProcessor:
    """
    Main class for processing PGA DFS data.
    Handles data loading, cleaning, imputation, and feature engineering.
    """

    def __init__(self, data_folder):
        """
        Initialize the data processor.

        Parameters:
        -----------
        data_folder : str
            Path to the folder containing the tournament data
        """
        self.data_folder = data_folder
        self.historical_dfs = None
        self.historical_rounds = None
        self.upcoming = None
        self.merged_data = None
        self.course_similarity_matrix = None

    def load_data(self):
        """Load all data files from the tournament folder."""
        print("Loading data files...")

        # Load historical DFS data
        self.historical_dfs = pd.read_csv(
            f"{self.data_folder}/historical_dfs_all.csv"
        )
        print(f"  - Loaded {len(self.historical_dfs):,} DFS records")

        # Load historical round data
        self.historical_rounds = pd.read_csv(
            f"{self.data_folder}/historical_round_all.csv"
        )
        print(f"  - Loaded {len(self.historical_rounds):,} round records")

        # Load upcoming tournament data
        self.upcoming = pd.read_csv(
            f"{self.data_folder}/upcoming.csv"
        )
        print(f"  - Loaded {len(self.upcoming):,} upcoming players")

        # Convert date columns
        self.historical_dfs['date'] = pd.to_datetime(self.historical_dfs['date'])
        self.historical_rounds['date'] = pd.to_datetime(self.historical_rounds['date'])

        return self

    def clean_data(self):
        """Clean and standardize the data."""
        print("\nCleaning data...")

        # Remove duplicates
        initial_dfs_count = len(self.historical_dfs)
        self.historical_dfs = self.historical_dfs.drop_duplicates()
        print(f"  - Removed {initial_dfs_count - len(self.historical_dfs)} duplicate DFS records")

        initial_round_count = len(self.historical_rounds)
        self.historical_rounds = self.historical_rounds.drop_duplicates()
        print(f"  - Removed {initial_round_count - len(self.historical_rounds)} duplicate round records")

        # Handle zero salaries in upcoming (if any)
        zero_salary_count = (self.upcoming['salary'] == 0).sum()
        if zero_salary_count > 0:
            print(f"  - Warning: {zero_salary_count} players with $0 salary (likely withdrawn)")
            self.upcoming = self.upcoming[self.upcoming['salary'] > 0]

        return self

    def impute_zeros_for_missed_cuts(self):
        """
        Add zero-score rows for rounds 3-4 when players missed the cut.
        This allows empirical sampling to naturally capture cut dynamics.

        Returns:
        --------
        self
        """
        print("\nImputing zeros for missed cuts...")

        # Count rounds per player per event
        rounds_per_event = (
            self.historical_rounds
            .groupby(['id_event', 'event_name', 'date', 'dg_id', 'player_name',
                      'course_name', 'id_course'])
            .agg({'round': lambda x: sorted(x.tolist())})
            .reset_index()
        )

        # Identify events where player missed cut (< 4 rounds)
        rounds_per_event['rounds_played'] = rounds_per_event['round'].apply(len)
        rounds_per_event['missed_cut'] = rounds_per_event['rounds_played'] < 4

        # Create zero rows for missing rounds
        zero_rows = []

        for idx, row in rounds_per_event[rounds_per_event['missed_cut']].iterrows():
            played_rounds = set(row['round'])
            all_rounds = {1.0, 2.0, 3.0, 4.0}
            missing_rounds = all_rounds - played_rounds

            for missing_round in missing_rounds:
                zero_row = {
                    'id_event': row['id_event'],
                    'event_name': row['event_name'],
                    'date': row['date'],
                    'dg_id': row['dg_id'],
                    'player_name': row['player_name'],
                    'course_name': row['course_name'],
                    'id_course': row['id_course'],
                    'round': missing_round,
                    'score': 0,  # Missed cut indicator
                    # Set all other stats to 0
                    'sg_total': 0,
                    'sg_putt': 0,
                    'sg_ott': 0,
                    'sg_app': 0,
                    'sg_arg': 0,
                    'birdies': 0,
                    'bogies': 0,
                    'pars': 0,
                    'eagles_or_better': 0,
                    'doubles_or_worse': 0,
                }
                zero_rows.append(zero_row)

        # Append zero rows to historical_rounds
        if zero_rows:
            zero_df = pd.DataFrame(zero_rows)
            self.historical_rounds = pd.concat(
                [self.historical_rounds, zero_df],
                ignore_index=True
            )

        print(f"  - Added {len(zero_rows):,} zero-score rows for missed cuts")
        print(f"  - Total rounds after imputation: {len(self.historical_rounds):,}")

        return self

    def aggregate_round_stats(self, groupby_cols=['dg_id', 'id_event'], last_n_rounds=None):
        """
        Aggregate round-level statistics to event level.

        Parameters:
        -----------
        groupby_cols : list
            Columns to group by (default: player + event)
        last_n_rounds : int, optional
            If provided, only use last N rounds for each player

        Returns:
        --------
        pd.DataFrame
            Aggregated statistics at the event level
        """
        print("\nAggregating round statistics...")

        df = self.historical_rounds.copy()

        # Define aggregation functions for different stat types
        stat_cols = {
            # Strokes gained metrics (mean)
            'sg_total': 'mean',
            'sg_t2g': 'mean',
            'sg_putt': 'mean',
            'sg_ott': 'mean',
            'sg_arg': 'mean',
            'sg_app': 'mean',

            # Scoring stats (mean and sum)
            'score': 'mean',
            'pars': 'sum',
            'birdies': 'sum',
            'bogies': 'sum',
            'doubles_or_worse': 'sum',
            'eagles_or_better': 'sum',

            # Shot quality (mean)
            'great_shots': 'mean',
            'poor_shots': 'mean',

            # Proximity and accuracy (mean)
            'prox_rgh': 'mean',
            'prox_fw': 'mean',
            'gir': 'mean',
            'driving_dist': 'mean',
            'driving_acc': 'mean',
            'scrambling': 'mean',
        }

        # Aggregate
        agg_dict = {}
        for col, agg_func in stat_cols.items():
            if col in df.columns:
                agg_dict[col] = agg_func

        # Add round count
        agg_dict['round'] = 'count'

        aggregated = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        aggregated = aggregated.rename(columns={'round': 'rounds_played'})

        print(f"  - Aggregated {len(df):,} rounds into {len(aggregated):,} event-player records")

        return aggregated

    def merge_historical_data(self):
        """Merge DFS and round data."""
        print("\nMerging historical datasets...")

        # Aggregate round stats first
        round_stats = self.aggregate_round_stats()

        # Merge with DFS data
        self.merged_data = self.historical_dfs.merge(
            round_stats,
            on=['dg_id', 'id_event'],
            how='left'
        )

        print(f"  - Created merged dataset with {len(self.merged_data):,} records")
        print(f"  - Features: {len(self.merged_data.columns)} columns")

        return self

    def create_player_features(self, lookback_events=5):
        """
        Create rolling player features based on recent performance.

        Parameters:
        -----------
        lookback_events : int
            Number of recent events to include in rolling calculations
        """
        print(f"\nCreating player features (lookback: {lookback_events} events)...")

        if self.merged_data is None:
            self.merge_historical_data()

        df = self.merged_data.copy()
        df = df.sort_values(['dg_id', 'date'])

        # Define features to roll
        roll_features = [
            'total_pts', 'sg_total', 'sg_putt', 'sg_ott', 'sg_arg', 'sg_app',
            'score', 'birdies', 'bogies', 'gir', 'driving_acc'
        ]

        # Create rolling stats for each player
        for feat in roll_features:
            if feat in df.columns:
                # Mean over last N events
                df[f'{feat}_L{lookback_events}_mean'] = (
                    df.groupby('dg_id')[feat]
                    .transform(lambda x: x.rolling(lookback_events, min_periods=1).mean().shift(1))
                )

                # Std over last N events
                df[f'{feat}_L{lookback_events}_std'] = (
                    df.groupby('dg_id')[feat]
                    .transform(lambda x: x.rolling(lookback_events, min_periods=1).std().shift(1))
                )

        # Add recent form (last 3 events)
        if lookback_events >= 3:
            df['recent_form_L3'] = (
                df.groupby('dg_id')['total_pts']
                .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            )

        self.merged_data = df

        print(f"  - Added rolling features for {len(roll_features)} base metrics")

        return self

    def impute_missing_values(self, method='knn', n_neighbors=5):
        """
        Impute missing values using specified method.

        Parameters:
        -----------
        method : str
            Imputation method ('knn', 'median', 'mean')
        n_neighbors : int
            Number of neighbors for KNN imputation
        """
        print(f"\nImputing missing values (method: {method})...")

        if self.merged_data is None:
            self.merge_historical_data()

        df = self.merged_data.copy()

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Count missing before
        missing_before = df[numeric_cols].isnull().sum().sum()
        print(f"  - Missing values before imputation: {missing_before:,}")

        if method == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        elif method == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == 'mice':
            # MICE requires clean integer index (drop=True avoids adding 'index' column)
            kernel = mf.ImputationKernel(df[numeric_cols].reset_index(drop=True), num_datasets=5, save_all_iterations_data=True, random_state=1842)
            kernel.mice(iterations=5, verbose=False)

            # Average across all 5 imputed datasets
            imputed_dfs = []
            for dataset in range(5):
                imputed_dfs.append(kernel.complete_data(dataset=dataset))

            # Average all datasets
            df[numeric_cols] = pd.concat(imputed_dfs).groupby(level=0).mean()
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        missing_after = df[numeric_cols].isnull().sum().sum()
        print(f"  - Missing values after imputation: {missing_after:,}")

        self.merged_data = df

        return self

    def build_course_similarity_matrix(self, features=None, min_coverage=0.5):
        """
        Build a course similarity matrix based on course characteristics.
        Only uses features where sufficient courses have actual data to avoid
        artificial similarity from imputed values.

        Uses course_name (not event_name) to properly handle multi-course events
        like Farmers Insurance Open (Torrey North + South).

        Parameters:
        -----------
        features : list, optional
            List of features to use for similarity calculation.
            If None, uses default set of course-relevant features.
        min_coverage : float
            Minimum fraction of courses that must have data for a feature
            to be included in similarity calculation (default: 0.5 = 50%)
        """
        print("\nBuilding course similarity matrix...")

        # Use historical_rounds directly (before event-level aggregation)
        # This preserves course-level granularity for multi-course events
        df = self.historical_rounds.copy()

        # Default features for course similarity (expanded to include more raw stats)
        if features is None:
            features = [
                # Raw scoring stats (almost always available)
                'score', 'birdies', 'bogies', 'pars',
                'doubles_or_worse', 'eagles_or_better',

                # Fundamental stats (usually available)
                'gir', 'driving_dist', 'driving_acc', 'scrambling',

                # Strokes gained (often missing for older/certain courses)
                'sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt'
            ]

        # Filter to available features in the dataset
        available_features = [f for f in features if f in df.columns]

        # Aggregate stats by COURSE (not event) to handle multi-course tournaments
        course_stats = df.groupby('course_name')[available_features].mean()

        # Calculate coverage for each feature (% of courses with data)
        coverage = course_stats.notna().sum() / len(course_stats)

        # Only keep features with sufficient coverage
        valid_features = coverage[coverage >= min_coverage].index.tolist()
        excluded_features = coverage[coverage < min_coverage].index.tolist()

        course_stats = course_stats[valid_features]

        # NOW fill NAs (only for features that passed coverage threshold)
        # This is acceptable because these features have sufficient real data
        course_stats = course_stats.fillna(course_stats.median())

        # Standardize features
        scaler = StandardScaler()
        course_stats_scaled = scaler.fit_transform(course_stats)

        # Calculate similarity matrix (1 - cosine distance)
        n_courses = len(course_stats)
        similarity_matrix = np.zeros((n_courses, n_courses))

        for i in range(n_courses):
            for j in range(n_courses):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = 1 - cosine(
                        course_stats_scaled[i],
                        course_stats_scaled[j]
                    )

        self.course_similarity_matrix = pd.DataFrame(
            similarity_matrix,
            index=course_stats.index,
            columns=course_stats.index
        )

        print(f"  - Created {n_courses}x{n_courses} similarity matrix")
        print(f"  - Using {len(valid_features)} features with >={min_coverage*100:.0f}% coverage:")
        print(f"    {', '.join(valid_features)}")
        if excluded_features:
            print(f"  - Excluded {len(excluded_features)} features due to low coverage:")
            print(f"    {', '.join(excluded_features)}")

        return self

    def get_similar_courses(self, target_course, top_n=5):
        """
        Get the most similar courses to a target course.

        Parameters:
        -----------
        target_course : str
            Name of the target course
        top_n : int
            Number of similar courses to return

        Returns:
        --------
        pd.Series
            Similarity scores for top N courses
        """
        if self.course_similarity_matrix is None:
            self.build_course_similarity_matrix()

        if target_course not in self.course_similarity_matrix.index:
            print(f"Warning: '{target_course}' not found in similarity matrix")
            return pd.Series()

        similarities = self.course_similarity_matrix[target_course].sort_values(ascending=False)

        # Exclude the course itself
        return similarities[1:top_n+1]

    def prepare_training_data(self, target_year=2025, features=None):
        """
        Prepare training and test datasets.

        Parameters:
        -----------
        target_year : int
            Year to use as test set (earlier years used for training)
        features : list, optional
            List of features to include in X. If None, uses default set.

        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test, feature_names)
        """
        print(f"\nPreparing training data (test year: {target_year})...")

        if self.merged_data is None:
            self.merge_historical_data()

        df = self.merged_data.copy()
        df['year'] = df['date'].dt.year

        # Default feature set
        if features is None:
            features = [
                # Rolling stats
                'total_pts_L5_mean', 'total_pts_L5_std',
                'sg_total_L5_mean', 'sg_total_L5_std',
                'sg_putt_L5_mean', 'sg_ott_L5_mean',
                'sg_arg_L5_mean', 'sg_app_L5_mean',

                # Recent stats
                'score_L5_mean', 'birdies_L5_mean',
                'gir_L5_mean', 'driving_acc_L5_mean',

                # Current salary (predictor of form)
                'salary',

                # Rounds played (consistency)
                'rounds_played'
            ]

        # Filter to available features
        available_features = [f for f in features if f in df.columns]

        # Split data
        train_df = df[df['year'] < target_year].copy()
        test_df = df[df['year'] == target_year].copy()

        # Remove rows with missing target
        train_df = train_df.dropna(subset=['total_pts'])
        test_df = test_df.dropna(subset=['total_pts'])

        # Prepare X and y
        X_train = train_df[available_features].fillna(0)
        y_train = train_df['total_pts']

        X_test = test_df[available_features].fillna(0)
        y_test = test_df['total_pts']

        print(f"  - Training set: {len(X_train):,} samples, {len(available_features)} features")
        print(f"  - Test set: {len(X_test):,} samples")
        print(f"  - Feature list: {', '.join(available_features[:5])}...")

        return X_train, y_train, X_test, y_test, available_features

    def get_upcoming_features(self, features=None):
        """
        Prepare feature matrix for upcoming tournament players.

        Parameters:
        -----------
        features : list, optional
            List of features to extract. If None, uses default set.

        Returns:
        --------
        pd.DataFrame
            Feature matrix for upcoming players with dg_id and player_name
        """
        print("\nPreparing upcoming player features...")

        if self.merged_data is None:
            self.merge_historical_data()

        # Get the most recent stats for each player
        df = self.merged_data.copy()
        df = df.sort_values(['dg_id', 'date'])
        latest_stats = df.groupby('dg_id').last().reset_index()

        # Drop columns from latest_stats that exist in upcoming to avoid _x and _y suffixes
        cols_to_drop = [col for col in ['player_name', 'salary'] if col in latest_stats.columns]
        if cols_to_drop:
            latest_stats = latest_stats.drop(columns=cols_to_drop)

        # Merge with upcoming tournament roster
        upcoming_features = self.upcoming.merge(
            latest_stats,
            on='dg_id',
            how='left'
        )

        print(f"  - Prepared features for {len(upcoming_features)} players")
        print(f"  - {upcoming_features['dg_id'].isin(latest_stats['dg_id']).sum()} players have historical data")

        return upcoming_features

    def get_course_adjustment_with_fallback(self, player_id, course_name, min_rounds=4):
        """
        Get course-specific scoring adjustment for a player with intelligent fallback
        for players with limited or no course history.

        Parameters:
        -----------
        player_id : int
            Player's dg_id
        course_name : str
            Name of the course
        min_rounds : int
            Minimum rounds needed to use direct course history

        Returns:
        --------
        tuple
            (adjustment, confidence) where adjustment is the scoring adjustment
            and confidence is a value between 0 and 1 indicating data quality
        """
        # Get player's rounds at this specific course
        player_course_rounds = self.historical_rounds[
            (self.historical_rounds['dg_id'] == player_id) &
            (self.historical_rounds['course_name'] == course_name)
        ]

        rounds_at_course = len(player_course_rounds)

        # Get player's overall scoring average
        player_all_rounds = self.historical_rounds[
            self.historical_rounds['dg_id'] == player_id
        ]

        if len(player_all_rounds) == 0:
            # No historical data at all
            return 0.0, 0.0

        overall_avg = player_all_rounds['score'].mean()

        if rounds_at_course >= min_rounds:
            # Sufficient data - use direct course history
            course_avg = player_course_rounds['score'].mean()
            adjustment = course_avg - overall_avg
            confidence = min(rounds_at_course / 20, 1.0)  # Cap at 20 rounds
            return adjustment, confidence

        elif rounds_at_course > 0:
            # Some data (1-3 rounds) - blend with similar courses
            course_avg = player_course_rounds['score'].mean()
            direct_adjustment = course_avg - overall_avg

            # Get similar course adjustment
            similar_adjustment, similar_confidence = self._get_similar_course_adjustment(
                player_id, course_name
            )

            # Blend based on sample size
            weight_direct = rounds_at_course / min_rounds
            adjustment = weight_direct * direct_adjustment + (1 - weight_direct) * similar_adjustment
            confidence = 0.3 + (0.3 * weight_direct)  # Lower confidence
            return adjustment, confidence

        else:
            # No course history - use similar courses only
            adjustment, confidence = self._get_similar_course_adjustment(player_id, course_name)
            return adjustment, confidence * 0.7  # Reduce confidence for fallback

    def _get_similar_course_adjustment(self, player_id, target_course, top_n=5):
        """
        Calculate scoring adjustment based on similar courses.

        Parameters:
        -----------
        player_id : int
            Player's dg_id
        target_course : str
            Name of the target course
        top_n : int
            Number of similar courses to consider

        Returns:
        --------
        tuple
            (adjustment, confidence)
        """
        # Build course similarity matrix if not already done
        if self.course_similarity_matrix is None:
            self.build_course_similarity_matrix()

        # Check if target course is in similarity matrix
        if target_course not in self.course_similarity_matrix.index:
            return 0.0, 0.2  # Low confidence for unknown course

        # Get most similar courses
        similar_courses = self.course_similarity_matrix[target_course].nlargest(top_n + 1).index[1:]

        adjustments = []
        weights = []

        player_all_rounds = self.historical_rounds[
            self.historical_rounds['dg_id'] == player_id
        ]

        if len(player_all_rounds) == 0:
            return 0.0, 0.0

        overall_avg = player_all_rounds['score'].mean()

        for similar_course in similar_courses:
            player_similar_rounds = self.historical_rounds[
                (self.historical_rounds['dg_id'] == player_id) &
                (self.historical_rounds['course_name'] == similar_course)
            ]

            if len(player_similar_rounds) > 0:
                course_avg = player_similar_rounds['score'].mean()
                adjustment = course_avg - overall_avg

                # Weight by similarity score and sample size
                similarity_weight = self.course_similarity_matrix.loc[target_course, similar_course]
                sample_weight = min(len(player_similar_rounds) / 10, 1.0)

                adjustments.append(adjustment)
                weights.append(similarity_weight * sample_weight)

        if len(adjustments) > 0:
            # Weighted average of similar course adjustments
            weighted_adj = np.average(adjustments, weights=weights)
            # Confidence based on how many similar courses we found
            confidence = min(len(adjustments) / top_n, 1.0) * 0.5  # Max 0.5 for similar courses
            return weighted_adj, confidence
        else:
            # No history at similar courses either
            return 0.0, 0.2

    def get_prediction_variance(self, player_id, course_name, base_variance):
        """
        Adjust prediction variance based on data availability for this player/course.

        Parameters:
        -----------
        player_id : int
            Player's dg_id
        course_name : str
            Name of the course
        base_variance : float
            Base variance from model

        Returns:
        --------
        float
            Adjusted variance
        """
        # Get rounds at this course
        player_course_rounds = self.historical_rounds[
            (self.historical_rounds['dg_id'] == player_id) &
            (self.historical_rounds['course_name'] == course_name)
        ]

        rounds_at_course = len(player_course_rounds)

        # Variance inflation factor based on sample size
        if rounds_at_course == 0:
            inflation = 1.5  # 50% more variance
        elif rounds_at_course < 4:
            inflation = 1.3
        elif rounds_at_course < 8:
            inflation = 1.15
        else:
            inflation = 1.0

        return base_variance * inflation

    def _sample_round_score(self, player_id, course_name, round_pred):
        """
        Sample a round score using empirical distribution when available.

        Parameters:
        -----------
        player_id : int
            FanDuel player ID
        course_name : str
            Name of the course
        round_pred : float
            Predicted round score from model

        Returns:
        --------
        int
            Sampled round score
        """
        # Get historical rounds at this course
        player_rounds = self.historical_rounds[
            (self.historical_rounds['dg_id'] == player_id) &
            (self.historical_rounds['course_name'] == course_name)
        ]['score'].values

        if len(player_rounds) >= 10:
            # Strategy 1: Empirical sampling centered on prediction
            historical_mean = np.mean(player_rounds)
            shift = round_pred - historical_mean
            sample = np.random.choice(player_rounds) + shift
            return round(sample)

        elif len(player_rounds) > 0:
            # Strategy 2: Blend empirical + normal
            weight = len(player_rounds) / 10
            empirical_sample = np.random.choice(player_rounds)
            normal_sample = np.random.normal(round_pred, 3.5)
            blended = weight * empirical_sample + (1 - weight) * normal_sample
            return round(blended)

        else:
            # Strategy 3: Normal fallback with variance inflation
            round_std = self.get_prediction_variance(player_id, course_name, 3.5)
            return round(np.random.normal(round_pred, round_std))

    def predict_with_round_simulation(self, upcoming_df, model, feature_names,
                                     event_schedule, n_simulations=1000):
        """
        Generate predictions using hybrid approach: event-level model + round-level simulation.

        Parameters:
        -----------
        upcoming_df : pd.DataFrame
            DataFrame with upcoming player features
        model : sklearn model
            Trained prediction model
        feature_names : list
            List of feature names used by model
        event_schedule : dict
            Event configuration with courses and rounds
        n_simulations : int
            Number of Monte Carlo simulations per player

        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with hybrid predictions and confidence intervals
        """
        print("\n" + "="*80)
        print("HYBRID ROUND-LEVEL SIMULATION")
        print("="*80)
        print(f"\nEvent: {event_schedule['event_name']} ({event_schedule['year']})")
        print(f"Courses:")
        for course_info in event_schedule['courses']:
            print(f"  - {course_info['name']}: Rounds {course_info['rounds']}")
        print(f"\nRunning {n_simulations:,} simulations per player...")

        results = []

        for idx, player in upcoming_df.iterrows():
            player_id = player['dg_id']
            player_name = player['player_name']

            # Step 1: Get event-level base prediction from model
            X_player = player[feature_names].fillna(0).values.reshape(1, -1)
            base_prediction = model.predict(X_player)[0]

            # Step 2: Run round-level simulations with smart zero handling
            simulated_totals = []
            cuts_made = 0
            cuts_missed = 0
            cut_after = event_schedule.get('cut_after', 2)  # Default to round 2

            for sim in range(n_simulations):
                event_total = 0
                missed_cut = False

                # Simulate each round in order [1, 2, 3, 4]
                for round_num in [1, 2, 3, 4]:
                    # Skip if already missed cut
                    if missed_cut:
                        continue  # All remaining rounds are 0

                    # Find course for this round
                    course_name = None
                    for course_info in event_schedule['courses']:
                        if round_num in course_info['rounds']:
                            course_name = course_info['name']
                            break

                    if course_name is None:
                        continue  # Round not in schedule

                    # Calculate round prediction
                    round_base = base_prediction / 4
                    adjustment, confidence = self.get_course_adjustment_with_fallback(
                        player_id, course_name, min_rounds=4
                    )
                    round_pred = round_base + adjustment

                    # Sample round score
                    round_actual = self._sample_round_score(player_id, course_name, round_pred)

                    # Handle zeros based on cut timing
                    if round_num <= cut_after:
                        # PRE-CUT: Zeros shouldn't happen (WD/DQ errors), resample
                        resample_count = 0
                        while round_actual == 0 and resample_count < 10:
                            round_actual = self._sample_round_score(player_id, course_name, round_pred)
                            resample_count += 1

                        if round_actual == 0:  # Still zero after retries
                            round_actual = round(round_pred)  # Fallback to prediction

                    elif round_num == cut_after + 1:
                        # FIRST POST-CUT: Zero = missed cut, cascade to remaining rounds
                        if round_actual == 0:
                            missed_cut = True
                            # Don't add to total, all future rounds will be skipped

                    else:
                        # LATER POST-CUT: Shouldn't be zero if we made cut, resample
                        resample_count = 0
                        while round_actual == 0 and resample_count < 10:
                            round_actual = self._sample_round_score(player_id, course_name, round_pred)
                            resample_count += 1

                        if round_actual == 0:  # Still zero after retries
                            round_actual = round(round_pred)  # Fallback to prediction

                    event_total += round_actual

                simulated_totals.append(event_total)

                # Track cut statistics
                if missed_cut:
                    cuts_missed += 1
                else:
                    cuts_made += 1

            # Calculate statistics
            simulated_array = np.array(simulated_totals)
            sim_cut_rate = cuts_made / n_simulations

            results.append({
                'dg_id': player_id,
                'player_name': player_name,
                'base_prediction': base_prediction,
                'hybrid_prediction': simulated_array.mean(),
                'hybrid_median': np.median(simulated_array),
                'hybrid_std': simulated_array.std(),
                'hybrid_p25': np.percentile(simulated_array, 25),
                'hybrid_p75': np.percentile(simulated_array, 75),
                'sim_cut_rate': sim_cut_rate,
                'cuts_made': cuts_made,
                'cuts_missed': cuts_missed,
                'confidence': simulated_array.std()  # Lower std = higher confidence
            })

        results_df = pd.DataFrame(results)

        # Merge back with original DataFrame
        upcoming_df = upcoming_df.merge(results_df[['dg_id', 'base_prediction', 'hybrid_prediction',
                                                     'hybrid_median', 'hybrid_std', 'hybrid_p25',
                                                     'hybrid_p75', 'sim_cut_rate', 'cuts_made',
                                                     'cuts_missed', 'confidence']],
                                       on='dg_id', how='left')

        print(f"\n✓ Completed {len(upcoming_df)} player predictions")
        print(f"  Average adjustment: {(upcoming_df['hybrid_prediction'] - upcoming_df['base_prediction']).mean():.2f} points")

        return upcoming_df


def optimize_lineup(predictions_df, salary_col='salary', points_col='predicted_pts',
                   budget=60000, lineup_size=6):
    """
    Optimize DFS lineup using linear programming.

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with player predictions, must include salary and predicted points
    salary_col : str
        Column name for player salary
    points_col : str
        Column name for predicted points
    budget : int
        Total salary budget
    lineup_size : int
        Number of players to select

    Returns:
    --------
    pd.DataFrame
        Optimal lineup with players and their stats
    """
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

    print("\n" + "="*60)
    print("OPTIMIZING LINEUP")
    print("="*60)

    df = predictions_df.copy()

    # Remove players with missing data
    df = df.dropna(subset=[salary_col, points_col])

    # Create the LP problem
    prob = LpProblem("DFS_Lineup_Optimizer", LpMaximize)

    # Decision variables (1 if player is selected, 0 otherwise)
    player_vars = []
    for idx in df.index:
        var = LpVariable(f"player_{idx}", cat='Binary')
        player_vars.append(var)

    # Objective function: maximize predicted points
    prob += lpSum([
        player_vars[i] * df.loc[idx, points_col]
        for i, idx in enumerate(df.index)
    ]), "Total_Points"

    # Constraint 1: Select exactly lineup_size players
    prob += lpSum(player_vars) == lineup_size, "Lineup_Size"

    # Constraint 2: Stay within budget
    prob += lpSum([
        player_vars[i] * df.loc[idx, salary_col]
        for i, idx in enumerate(df.index)
    ]) <= budget, "Salary_Cap"

    # Solve
    prob.solve()

    # Extract solution
    selected_indices = [idx for i, idx in enumerate(df.index) if player_vars[i].varValue == 1]
    lineup = df.loc[selected_indices].copy()

    # Print results
    print(f"\nStatus: {LpStatus[prob.status]}")
    print(f"Total Projected Points: {lineup[points_col].sum():.1f}")
    print(f"Total Salary: ${lineup[salary_col].sum():,.0f} / ${budget:,.0f}")
    print(f"Remaining Salary: ${budget - lineup[salary_col].sum():,.0f}")
    print("\nLineup:")
    print("-" * 60)

    lineup_display = lineup[['player_name', salary_col, points_col]].copy()
    lineup_display = lineup_display.sort_values(points_col, ascending=False)

    for idx, row in lineup_display.iterrows():
        print(f"{row['player_name']:25s} ${row[salary_col]:>6,.0f}  {row[points_col]:>6.1f} pts")

    print("="*60 + "\n")

    return lineup

#possibly not needed anymore
def monte_carlo_simulation(predictions_df, n_simulations=1000,
                          salary_col='salary', points_col='predicted_pts',
                          std_col='predicted_std', budget=60000, lineup_size=6):
    """
    Run Monte Carlo simulation to assess lineup variance.

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with player predictions and uncertainty estimates
    n_simulations : int
        Number of simulations to run
    salary_col : str
        Column name for player salary
    points_col : str
        Column name for predicted points
    std_col : str
        Column name for prediction standard deviation
    budget : int
        Total salary budget
    lineup_size : int
        Number of players to select

    Returns:
    --------
    dict
        Dictionary containing simulation results and statistics
    """
    print("\n" + "="*60)
    print(f"MONTE CARLO SIMULATION ({n_simulations:,} iterations)")
    print("="*60)

    # Get baseline optimal lineup
    baseline_lineup = optimize_lineup(
        predictions_df, salary_col, points_col, budget, lineup_size
    )

    # Initialize results storage
    simulation_results = []

    # Run simulations
    for sim in range(n_simulations):
        # Generate simulated points for each player
        sim_df = predictions_df.copy()

        # If std provided, use it; otherwise use 15% of predicted points as std
        if std_col in sim_df.columns:
            sim_df['sim_pts'] = np.random.normal(
                sim_df[points_col],
                sim_df[std_col]
            )
        else:
            sim_df['sim_pts'] = np.random.normal(
                sim_df[points_col],
                sim_df[points_col] * 0.15
            )

        # Ensure non-negative points
        sim_df['sim_pts'] = sim_df['sim_pts'].clip(lower=0)

        # Calculate total points for baseline lineup in this simulation
        baseline_ids = baseline_lineup.index
        sim_total = sim_df.loc[baseline_ids, 'sim_pts'].sum()

        simulation_results.append(sim_total)

    # Calculate statistics
    results_array = np.array(simulation_results)

    stats = {
        'mean': results_array.mean(),
        'median': np.median(results_array),
        'std': results_array.std(),
        'min': results_array.min(),
        'max': results_array.max(),
        'percentile_25': np.percentile(results_array, 25),
        'percentile_75': np.percentile(results_array, 75),
        'simulations': simulation_results,
        'baseline_lineup': baseline_lineup
    }

    print(f"\nSimulation Results:")
    print(f"  Mean Score: {stats['mean']:.1f}")
    print(f"  Median Score: {stats['median']:.1f}")
    print(f"  Std Dev: {stats['std']:.1f}")
    print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
    print(f"  25th-75th Percentile: {stats['percentile_25']:.1f} - {stats['percentile_75']:.1f}")
    print("="*60 + "\n")

    return stats


def get_player_historical_scores(processor, min_events=5, recency_weight=True):
    """
    Get historical DFS scores for each player to build empirical distributions.

    Parameters:
    -----------
    processor : DFSDataProcessor
        The data processor with merged historical data
    min_events : int
        Minimum number of events required for a player
    recency_weight : bool
        Whether to weight recent events more heavily

    Returns:
    --------
    dict
        Dictionary mapping dg_id to list of historical scores
    """
    print("\nBuilding empirical score distributions...")

    if processor.merged_data is None:
        processor.merge_historical_data()

    df = processor.merged_data.copy()
    df = df.sort_values(['dg_id', 'date'])

    player_scores = {}
    players_with_history = 0

    for dg_id in df['dg_id'].unique():
        player_data = df[df['dg_id'] == dg_id]['total_pts'].dropna()

        if len(player_data) >= min_events:
            scores = player_data.values

            # Apply recency weighting if requested
            if recency_weight and len(scores) > 5:
                # Weight: last 5 events get 2x weight, rest get 1x
                recent = scores[-5:]
                older = scores[:-5]
                weighted_scores = list(older) + list(recent) * 2
                player_scores[dg_id] = weighted_scores
            else:
                player_scores[dg_id] = list(scores)

            players_with_history += 1

    print(f"  - Built distributions for {players_with_history} players")
    print(f"  - Average events per player: {np.mean([len(s) for s in player_scores.values()]):.1f}")

    return player_scores


def monte_carlo_empirical(lineup_df, player_scores, n_simulations=10000,
                         fallback_std=20, dg_id_col='dg_id'):
    """
    Run Monte Carlo simulation using empirical distributions from historical scores.

    Parameters:
    -----------
    lineup_df : pd.DataFrame
        DataFrame with the lineup players
    player_scores : dict
        Dictionary mapping dg_id to list of historical scores
    n_simulations : int
        Number of simulations to run
    fallback_std : float
        Standard deviation to use for players without history
    dg_id_col : str
        Column name for player ID

    Returns:
    --------
    dict
        Dictionary with simulation results and advanced metrics
    """
    simulation_results = []

    for sim in range(n_simulations):
        lineup_total = 0

        for idx, player in lineup_df.iterrows():
            dg_id = player[dg_id_col]

            if dg_id in player_scores and len(player_scores[dg_id]) > 0:
                # Sample from player's empirical distribution
                score = np.random.choice(player_scores[dg_id])
            else:
                # Fallback to normal distribution for players without history
                predicted = player.get('predicted_pts', 70)
                score = np.random.normal(predicted, fallback_std)
                score = max(0, score)  # Ensure non-negative

            lineup_total += score

        simulation_results.append(lineup_total)

    # Calculate comprehensive statistics
    results_array = np.array(simulation_results)

    from scipy import stats as scipy_stats

    stats = {
        'simulations': simulation_results,
        'mean': results_array.mean(),
        'median': np.median(results_array),
        'std': results_array.std(),
        'min': results_array.min(),
        'max': results_array.max(),
        'percentile_5': np.percentile(results_array, 5),
        'percentile_10': np.percentile(results_array, 10),
        'percentile_25': np.percentile(results_array, 25),
        'percentile_75': np.percentile(results_array, 75),
        'percentile_90': np.percentile(results_array, 90),
        'percentile_95': np.percentile(results_array, 95),
        'skewness': scipy_stats.skew(results_array),
        'kurtosis': scipy_stats.kurtosis(results_array),
        'cv': results_array.std() / results_array.mean() if results_array.mean() > 0 else 0,  # Coefficient of variation
    }

    # Sharpe-like ratio (using median as baseline)
    baseline = np.median(results_array)
    stats['sharpe_ratio'] = (stats['mean'] - baseline) / stats['std'] if stats['std'] > 0 else 0

    return stats


def generate_diverse_lineups(predictions_df, player_scores, n_lineups=20,
                            salary_col='salary', points_col='predicted_pts',
                            budget=60000, lineup_size=6):
    """
    Generate diverse lineups using multiple strategies.

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with player predictions
    player_scores : dict
        Historical scores for empirical distributions
    n_lineups : int
        Number of diverse lineups to generate
    salary_col : str
        Salary column name
    points_col : str
        Predicted points column name
    budget : int
        Salary budget
    lineup_size : int
        Number of players per lineup

    Returns:
    --------
    list
        List of dictionaries, each containing lineup DataFrame and metadata
    """
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, LpBinary

    print("\n" + "="*80)
    print(f"GENERATING {n_lineups} DIVERSE LINEUPS")
    print("="*80 + "\n")

    lineups = []
    df = predictions_df.copy()
    df = df.dropna(subset=[salary_col, points_col])

    # Strategy 1: Optimal lineup
    print("1. Generating optimal lineup...")
    optimal = optimize_lineup(df, salary_col, points_col, budget, lineup_size)
    lineups.append({
        'lineup': optimal,
        'strategy': 'Optimal',
        'description': 'Maximum predicted points'
    })

    # Strategy 2-6: Sequential exclusion by salary (contrarian)
    print("2-6. Generating salary-based exclusions...")
    for i in range(1, 6):
        top_n_salary = df.nlargest(i, salary_col).index
        excluded_df = df.drop(top_n_salary)

        lineup = optimize_lineup(excluded_df, salary_col, points_col, budget, lineup_size)
        lineups.append({
            'lineup': lineup,
            'strategy': f'Exclude_Top{i}_Salary',
            'description': f'Exclude top {i} by salary'
        })

    # Strategy 7-11: Sequential exclusion by predicted points
    print("7-11. Generating prediction-based exclusions...")
    for i in range(1, 6):
        top_n_pred = df.nlargest(i, points_col).index
        excluded_df = df.drop(top_n_pred)

        lineup = optimize_lineup(excluded_df, salary_col, points_col, budget, lineup_size)
        lineups.append({
            'lineup': lineup,
            'strategy': f'Exclude_Top{i}_Predicted',
            'description': f'Exclude top {i} by prediction'
        })

    # Strategy 12-14: Random prediction perturbations
    print("12-14. Generating prediction perturbation lineups...")
    for i in range(3):
        perturbed_df = df.copy()
        # Add small random noise to predictions to explore solution space
        noise_std = df[points_col].std() * 0.1  # 10% of prediction std
        perturbed_df['perturbed_pts'] = perturbed_df[points_col] + np.random.normal(0, noise_std, len(perturbed_df))

        lineup = optimize_lineup(perturbed_df, salary_col, 'perturbed_pts', budget, lineup_size)
        # Restore original data
        lineup = df.loc[lineup.index]
        lineups.append({
            'lineup': lineup,
            'strategy': f'Perturbed_{i+1}',
            'description': f'Prediction perturbation (variation {i+1})'
        })

    # Strategy 15-17: Variance-based (using historical score variance)
    print("15-17. Generating variance-based lineups...")
    df['score_std'] = df['dg_id'].map(lambda x: np.std(player_scores.get(x, [70])) if x in player_scores else 20)

    # High variance (boom or bust)
    high_var_df = df.nlargest(50, 'score_std')
    lineup = optimize_lineup(high_var_df, salary_col, points_col, budget, lineup_size)
    lineups.append({
        'lineup': lineup,
        'strategy': 'High_Variance',
        'description': 'High variance players (boom/bust)'
    })

    # Low variance (consistent)
    low_var_df = df.nsmallest(50, 'score_std')
    lineup = optimize_lineup(low_var_df, salary_col, points_col, budget, lineup_size)
    lineups.append({
        'lineup': lineup,
        'strategy': 'Low_Variance',
        'description': 'Low variance players (consistent)'
    })

    # Medium variance (balanced)
    df['var_rank'] = df['score_std'].rank(pct=True)
    balanced_df = df[(df['var_rank'] > 0.3) & (df['var_rank'] < 0.7)]
    if len(balanced_df) >= lineup_size:
        lineup = optimize_lineup(balanced_df, salary_col, points_col, budget, lineup_size)
        lineups.append({
            'lineup': lineup,
            'strategy': 'Balanced_Variance',
            'description': 'Medium variance players (balanced)'
        })

    print(f"\n✓ Generated {len(lineups)} diverse lineups")
    print("="*80 + "\n")

    return lineups


def compare_lineups(lineups, player_scores, n_simulations=10000, tournament_type='gpp'):
    """
    Compare multiple lineups using empirical Monte Carlo simulation.

    Parameters:
    -----------
    lineups : list
        List of lineup dictionaries from generate_diverse_lineups()
    player_scores : dict
        Historical scores for empirical distributions
    n_simulations : int
        Number of Monte Carlo simulations per lineup
    tournament_type : str
        'gpp' (maximize upside) or 'cash' (minimize risk)

    Returns:
    --------
    pd.DataFrame
        Comparison table with all lineups ranked
    """
    print("\n" + "="*80)
    print(f"COMPARING {len(lineups)} LINEUPS WITH MONTE CARLO")
    print(f"Simulations per lineup: {n_simulations:,}")
    print(f"Tournament type: {tournament_type.upper()}")
    print("="*80 + "\n")

    results = []

    for i, lineup_dict in enumerate(lineups):
        print(f"[{i+1}/{len(lineups)}] Simulating {lineup_dict['strategy']}...", end=' ')

        lineup_df = lineup_dict['lineup']

        # Run empirical Monte Carlo
        mc_stats = monte_carlo_empirical(lineup_df, player_scores, n_simulations)

        # Compile results
        result = {
            'Rank': 0,  # Will be assigned later
            'Strategy': lineup_dict['strategy'],
            'Description': lineup_dict['description'],
            'Total_Salary': lineup_df['salary'].sum(),
            'Predicted_Pts': lineup_df['predicted_pts'].sum(),
            'MC_Mean': mc_stats['mean'],
            'MC_Median': mc_stats['median'],
            'MC_Std': mc_stats['std'],
            'MC_P75': mc_stats['percentile_75'],
            'MC_P90': mc_stats['percentile_90'],
            'MC_P95': mc_stats['percentile_95'],
            'Skewness': mc_stats['skewness'],
            'Sharpe': mc_stats['sharpe_ratio'],
            'CV': mc_stats['cv'],
            'lineup_obj': lineup_df,
            'mc_stats': mc_stats
        }

        results.append(result)
        print(f"✓ Mean: {mc_stats['mean']:.1f}, P75: {mc_stats['percentile_75']:.1f}")

    # Create comparison DataFrame
    comparison = pd.DataFrame(results)

    # Rank based on tournament type
    if tournament_type.lower() == 'gpp':
        # For GPP: maximize 75th percentile (upside potential)
        comparison = comparison.sort_values('MC_P75', ascending=False)
        ranking_metric = 'MC_P75'
    else:  # cash games
        # For cash: maximize median, minimize variance
        comparison['Cash_Score'] = comparison['MC_Median'] - comparison['MC_Std']
        comparison = comparison.sort_values('Cash_Score', ascending=False)
        ranking_metric = 'Cash_Score'

    comparison['Rank'] = range(1, len(comparison) + 1)

    # Display top 10
    print("\n" + "="*80)
    print(f"TOP 10 LINEUPS (Ranked by {ranking_metric})")
    print("="*80 + "\n")

    display_cols = ['Rank', 'Strategy', 'Total_Salary', 'MC_Mean', 'MC_Median',
                   'MC_P75', 'MC_P95', 'MC_Std', 'Skewness']
    print(comparison[display_cols].head(10).to_string(index=False))
    print("\n" + "="*80 + "\n")

    return comparison


def simulate_event_with_cuts(field_players, target_course, train_rounds,
                              course_similarity_matrix, event_schedule,
                              n_simulations=1000):
    """
    Simulate full tournament with competitive cut line using pure empirical sampling.

    Uses course-weighted blending to sample rounds, simulates cuts based on
    field-wide R1-R2 performance, and calculates DFS points.

    Parameters:
    -----------
    field_players : list
        List of player IDs in the field
    target_course : str
        Course name (e.g., 'TPC Scottsdale')
    train_rounds : pd.DataFrame
        Historical training data (pre-2025)
    course_similarity_matrix : pd.DataFrame
        Course similarity matrix
    event_schedule : dict
        Event configuration:
        - 'cut_after': 2 (cut after round 2)
        - 'cut_proportion': 0.46 (default, ~46% make cut)
    n_simulations : int
        Number of Monte Carlo simulations (default: 1000)

    Returns:
    --------
    predictions : pd.DataFrame
        Columns: dg_id, mean_dfs_pts, median_dfs_pts, std_dfs_pts, p25_dfs_pts,
                 p75_dfs_pts, cut_rate
    """
    results = {player_id: [] for player_id in field_players}
    cuts_made = {player_id: 0 for player_id in field_players}

    # Calculate cut position from field size and proportion
    cut_proportion = event_schedule.get('cut_proportion', 0.46)
    cut_position = int(len(field_players) * cut_proportion)

    for sim in range(n_simulations):
        # Phase 1: Simulate R1-R2 for ALL players
        r1_r2_totals = {}

        for player_id in field_players:
            # Sample R1-R2 from historical data (course-weighted)
            rounds_r1_r2, blend_info = get_course_weighted_samples(
                player_id, target_course, train_rounds,
                course_similarity_matrix, n_samples=2, min_rounds=10
            )

            if rounds_r1_r2 is None or blend_info['insufficient_data']:
                # Insufficient data - assume missed cut
                r1_r2_totals[player_id] = 0
            else:
                # Convert golf stats to hole_score_pts
                r1_r2_pts = calculate_hole_score_pts(rounds_r1_r2)
                r1_r2_totals[player_id] = r1_r2_pts.sum()

        # Phase 2: Calculate competitive cut line (discrete score cutoff)
        sorted_totals = sorted(r1_r2_totals.items(), key=lambda x: -x[1])
        cut_line_score = sorted_totals[cut_position - 1][1] if len(sorted_totals) >= cut_position else 0

        # Phase 3: Simulate R3-R4 for players who made cut
        for player_id in field_players:
            r1_r2_score = r1_r2_totals[player_id]

            if r1_r2_score >= cut_line_score and r1_r2_score > 0:
                # Made cut - sample R3-R4
                cuts_made[player_id] += 1

                rounds_r3_r4, _ = get_course_weighted_samples(
                    player_id, target_course, train_rounds,
                    course_similarity_matrix, n_samples=2, min_rounds=10
                )

                if rounds_r3_r4 is not None:
                    r3_r4_pts = calculate_hole_score_pts(rounds_r3_r4)
                    total_score = r1_r2_score + r3_r4_pts.sum()
                else:
                    # Fallback if R3-R4 sampling fails
                    total_score = r1_r2_score
            else:
                # Missed cut - only R1-R2 counts
                total_score = r1_r2_score

            results[player_id].append(total_score)

    # Aggregate results across simulations
    predictions = []
    for player_id in field_players:
        sim_array = np.array(results[player_id])
        predictions.append({
            'dg_id': player_id,
            'mean_dfs_pts': sim_array.mean(),
            'median_dfs_pts': np.median(sim_array),
            'std_dfs_pts': sim_array.std(),
            'p25_dfs_pts': np.percentile(sim_array, 25),
            'p75_dfs_pts': np.percentile(sim_array, 75),
            'cut_rate': cuts_made[player_id] / n_simulations
        })

    return pd.DataFrame(predictions)


def get_course_weighted_samples(player_id, target_course, train_rounds,
                                 course_similarity_matrix, n_samples=4,
                                 top_k_similar=5, min_rounds=10):
    """
    Sample rounds for a player with automatic blending based on data availability.

    Uses course-weighted blending: samples from specific course + similar courses
    based on data availability ratio.

    Parameters:
    -----------
    player_id : int
        Player DG ID
    target_course : str
        Course name to predict for
    train_rounds : pd.DataFrame
        Historical training data (pre-2025)
    course_similarity_matrix : pd.DataFrame
        Course similarity scores
    n_samples : int
        Number of rounds to sample (typically 2 for R1-R2, 2 for R3-R4)
    top_k_similar : int
        Number of similar courses to include (default: 5)
    min_rounds : int
        Minimum total rounds needed (default: 10)

    Returns:
    --------
    sampled_rounds : pd.DataFrame or None
        Sampled rounds with birdies, eagles_or_better, pars, bogies, doubles_or_worse
        Returns None if insufficient data (< min_rounds)
    blend_info : dict
        {'n_specific': int, 'n_similar': int, 'weight_specific': float, 'insufficient_data': bool}
    """
    # Get specific course rounds
    specific_rounds = train_rounds[
        (train_rounds['dg_id'] == player_id) &
        (train_rounds['course_name'] == target_course)
    ]
    n_specific = len(specific_rounds)

    # Get top K similar course rounds
    similar_courses = []
    if target_course in course_similarity_matrix.index:
        similar_courses = (
            course_similarity_matrix.loc[target_course]
            .sort_values(ascending=False)
            .iloc[1:top_k_similar+1]  # Exclude self, take top K
            .index.tolist()
        )

    similar_rounds = train_rounds[
        (train_rounds['dg_id'] == player_id) &
        (train_rounds['course_name'].isin(similar_courses))
    ]
    n_similar = len(similar_rounds)

    # Check minimum data threshold
    if n_specific + n_similar < min_rounds:
        # Insufficient data - player will be assumed to miss cut
        return None, {
            'n_specific': n_specific,
            'n_similar': n_similar,
            'weight_specific': 0,
            'insufficient_data': True
        }

    # Calculate blend weight
    weight_specific = n_specific / (n_specific + n_similar)

    # Sample with replacement based on blend weight
    n_specific_samples = int(n_samples * weight_specific)
    n_similar_samples = n_samples - n_specific_samples

    sampled_specific = specific_rounds.sample(
        n=n_specific_samples,
        replace=True
    ) if n_specific > 0 and n_specific_samples > 0 else pd.DataFrame()

    sampled_similar = similar_rounds.sample(
        n=n_similar_samples,
        replace=True
    ) if n_similar > 0 and n_similar_samples > 0 else pd.DataFrame()

    sampled_rounds = pd.concat([sampled_specific, sampled_similar], ignore_index=True)

    blend_info = {
        'n_specific': n_specific,
        'n_similar': n_similar,
        'weight_specific': weight_specific,
        'insufficient_data': False
    }

    return sampled_rounds, blend_info


def calculate_hole_score_pts(rounds_df):
    """
    Convert round-level golf statistics to DFS hole_score_pts.

    Uses FanDuel scoring (eagles, birdies, pars, bogeys, doubles).
    Does NOT include: finish_pts, streak_pts, bounce_back_pts, etc.

    Parameters:
    -----------
    rounds_df : pd.DataFrame
        Rounds with eagles_or_better, birdies, pars, bogies, doubles_or_worse columns

    Returns:
    --------
    hole_score_pts : np.array
        DFS hole scoring points per round
    """
    points = (
        rounds_df['eagles_or_better'] * 7 +
        rounds_df['birdies'] * 3.1 +
        rounds_df['pars'] * 0.5 -
        rounds_df['bogies'] * 1 -
        rounds_df['doubles_or_worse'] * 3
    )

    return points.values


if __name__ == "__main__":
    print("DFS Data Utilities Module")
    print("This module provides functions for PGA DFS data processing and optimization.")
