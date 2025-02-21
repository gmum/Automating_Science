"""
# Notebook 1: Foundations of Scientific AutoML and Data Processing
## A Comprehensive Introduction to Automated Scientific Discovery

### Course Overview
This notebook is the first in a series of five focusing on automating scientific processes using machine learning. 
We'll cover fundamental concepts and build practical skills for automated scientific data analysis.

### Learning Objectives
By the end of this notebook, you will be able to:
- Implement automated data processing pipelines for scientific data
- Apply AutoML techniques to scientific problems
- Develop automated feature engineering systems
- Create validation frameworks for scientific ML models
- Generate comprehensive analysis reports automatically
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tpot import TPOTRegressor
from autosklearn.regression import AutoSklearnRegressor
import optuna
import warnings
from IPython.display import display, HTML
import joblib

# Visualization settings
plt.style.use('seaborn')
sns.set_context("talk")
warnings.filterwarnings('ignore')

"""
## Section 1: Scientific Data Generation and Analysis

We'll start by creating synthetic scientific data that mimics real-world scenarios.
This will help us understand common challenges in scientific data analysis.
"""

class ScientificDataGenerator:
    def __init__(self, noise_level=0.02):
        self.noise_level = noise_level
        
    def generate_spectroscopy_data(self, n_samples=1000, n_wavelengths=100):
        """Generate synthetic spectroscopy data with multiple peaks and noise"""
        wavelengths = np.linspace(400, 800, n_wavelengths)
        spectra = []
        concentrations = []
        
        for _ in range(n_samples):
            # Generate random concentrations for different components
            c1, c2, c3 = np.random.uniform(0.1, 1.0, 3)
            concentrations.append([c1, c2, c3])
            
            # Generate spectrum with multiple peaks
            spectrum = (
                c1 * self._generate_peak(wavelengths, 550, 50) +
                c2 * self._generate_peak(wavelengths, 650, 30) +
                c3 * self._generate_peak(wavelengths, 450, 40) +
                np.random.normal(0, self.noise_level, n_wavelengths)
            )
            spectra.append(spectrum)
            
        # Create DataFrame with wavelength columns
        spectra_df = pd.DataFrame(
            spectra,
            columns=[f'wl_{w:.0f}' for w in wavelengths]
        )
        
        # Create concentration DataFrame
        conc_df = pd.DataFrame(
            concentrations,
            columns=['Component_A', 'Component_B', 'Component_C']
        )
        
        return spectra_df, conc_df
    
    def generate_time_series_data(self, n_samples=1000, n_timepoints=100):
        """Generate synthetic time series data with different patterns"""
        time = np.linspace(0, 10, n_timepoints)
        data = []
        parameters = []
        
        for _ in range(n_samples):
            # Random parameters for the time series
            freq = np.random.uniform(0.5, 2.0)
            amp = np.random.uniform(0.5, 2.0)
            decay = np.random.uniform(0.1, 0.5)
            
            # Generate complex time series
            series = (
                amp * np.sin(2 * np.pi * freq * time) * np.exp(-decay * time) +
                np.random.normal(0, self.noise_level, n_timepoints)
            )
            data.append(series)
            parameters.append([freq, amp, decay])
            
        # Create DataFrames
        time_series_df = pd.DataFrame(
            data,
            columns=[f't_{t:.2f}' for t in time]
        )
        param_df = pd.DataFrame(
            parameters,
            columns=['Frequency', 'Amplitude', 'Decay_Rate']
        )
        
        return time_series_df, param_df
    
    def _generate_peak(self, x, center, width):
        """Generate a Gaussian peak"""
        return np.exp(-(x - center)**2 / (2 * width**2))

# Generate example datasets
data_gen = ScientificDataGenerator()
spectral_data, concentrations = data_gen.generate_spectroscopy_data()
time_series_data, parameters = data_gen.generate_time_series_data()

"""
## Section 2: Advanced Data Processing Pipeline

We'll create a comprehensive data processing pipeline that handles various types of scientific data.
"""

class AdvancedDataProcessor:
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        
    def analyze_data_quality(self, df):
        """Comprehensive data quality analysis"""
        report = {
            'basic_stats': df.describe(),
            'missing_values': {
                'count': df.isnull().sum(),
                'percentage': (df.isnull().sum() / len(df) * 100).round(2)
            },
            'unique_values': df.nunique(),
            'data_types': df.dtypes,
            'correlations': df.corr(),
            'skewness': df.skew(),
            'kurtosis': df.kurtosis()
        }
        
        # Visualize distributions
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(df.columns[:3]):  # First 3 columns as example
            plt.subplot(1, 3, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        
        return report
    
    def preprocess_data(self, df, scaler_type='standard', n_components=None,
                       feature_selection='pca', target=None):
        """
        Advanced preprocessing pipeline with multiple options
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        scaler_type : str
            Type of scaling ('standard', 'robust', or 'minmax')
        n_components : int
            Number of components for dimensionality reduction
        feature_selection : str
            Method for feature selection ('pca' or 'mutual_info')
        target : array-like
            Target variable for supervised feature selection
        """
        # Handle missing values
        df_clean = self._handle_missing_values(df)
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scalers[scaler_type].fit_transform(df_clean),
            columns=df_clean.columns
        )
        
        # Feature selection/dimensionality reduction
        if feature_selection == 'pca':
            if n_components is None:
                n_components = min(len(df_scaled.columns), len(df_scaled))
            self.pca = PCA(n_components=n_components)
            transformed_data = self.pca.fit_transform(df_scaled)
            df_transformed = pd.DataFrame(
                transformed_data,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
        elif feature_selection == 'mutual_info' and target is not None:
            self.feature_selector = SelectKBest(
                mutual_info_regression,
                k=n_components or len(df_scaled.columns)
            )
            transformed_data = self.feature_selector.fit_transform(df_scaled, target)
            selected_features = df_scaled.columns[self.feature_selector.get_support()]
            df_transformed = pd.DataFrame(transformed_data, columns=selected_features)
            
        else:
            df_transformed = df_scaled
            
        return df_transformed
    
    def _handle_missing_values(self, df):
        """Advanced missing value handling"""
        df_clean = df.copy()
        
        # For each column, choose appropriate imputation method
        for column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            if missing_count > 0:
                if missing_count / len(df_clean) < 0.05:  # Less than 5% missing
                    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                else:
                    # Use more sophisticated imputation for columns with more missing values
                    df_clean[column] = df_clean[column].fillna(
                        df_clean[column].interpolate(method='cubic')
                    )
        
        return df_clean

"""
## Section 3: Advanced Feature Engineering

We'll implement sophisticated feature engineering specifically designed for scientific data.
"""

class ScientificFeatureEngineer:
    def __init__(self):
        self.feature_history = {}
        
    def engineer_spectral_features(self, spectra_df):
        """Engineer features for spectral data"""
        features = pd.DataFrame(index=spectra_df.index)
        
        # Basic statistical features
        features['max_intensity'] = spectra_df.max(axis=1)
        features['mean_intensity'] = spectra_df.mean(axis=1)
        features['std_intensity'] = spectra_df.std(axis=1)
        features['median_intensity'] = spectra_df.median(axis=1)
        features['skewness'] = spectra_df.skew(axis=1)
        features['kurtosis'] = spectra_df.kurtosis(axis=1)
        
        # Peak analysis
        features = pd.concat([
            features,
            self._analyze_peaks(spectra_df)
        ], axis=1)
        
        # Derivative features
        features = pd.concat([
            features,
            self._calculate_derivatives(spectra_df)
        ], axis=1)
        
        # Area features
        features = pd.concat([
            features,
            self._calculate_areas(spectra_df)
        ], axis=1)
        
        self.feature_history['spectral'] = features.columns.tolist()
        return features
    
    def engineer_time_series_features(self, time_series_df):
        """Engineer features for time series data"""
        features = pd.DataFrame(index=time_series_df.index)
        
        # Statistical features
        features['mean'] = time_series_df.mean(axis=1)
        features['std'] = time_series_df.std(axis=1)
        features['max'] = time_series_df.max(axis=1)
        features['min'] = time_series_df.min(axis=1)
        
        # Frequency domain features
        features = pd.concat([
            features,
            self._analyze_frequency_domain(time_series_df)
        ], axis=1)
        
        # Trend features
        features = pd.concat([
            features,
            self._analyze_trends(time_series_df)
        ], axis=1)
        
        self.feature_history['time_series'] = features.columns.tolist()
        return features
    
    def _analyze_peaks(self, spectra_df):
        """Analyze peaks in spectral data"""
        peak_features = pd.DataFrame(index=spectra_df.index)
        
        for idx, spectrum in spectra_df.iterrows():
            peaks, properties = signal.find_peaks(
                spectrum,
                height=0.1,
                distance=5,
                prominence=0.1
            )
            
            peak_features.at[idx, 'n_peaks'] = len(peaks)
            if len(peaks) > 0:
                peak_features.at[idx, 'max_peak_height'] = max(properties['peak_heights'])
                peak_features.at[idx, 'mean_peak_height'] = np.mean(properties['peak_heights'])
                peak_features.at[idx, 'peak_spacing'] = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
            else:
                peak_features.at[idx, 'max_peak_height'] = 0
                peak_features.at[idx, 'mean_peak_height'] = 0
                peak_features.at[idx, 'peak_spacing'] = 0
                
        return peak_features
    
    def _calculate_derivatives(self, df):
        """Calculate first and second derivatives"""
        derivative_features = pd.DataFrame(index=df.index)
        
        # First derivative
        first_deriv = np.gradient(df.values, axis=1)
        derivative_features['max_first_deriv'] = np.max(first_deriv, axis=1)
        derivative_features['min_first_deriv'] = np.min(first_deriv, axis=1)
        derivative_features['mean_first_deriv'] = np.mean(first_deriv, axis=1)
        
        # Second derivative
        second_deriv = np.gradient(first_deriv, axis=1)
        derivative_features['max_second_deriv'] = np.max(second_deriv, axis=1)
        derivative_features['min_second_deriv'] = np.min(second_deriv, axis=1)
        derivative_features['mean_second_deriv'] = np.mean(second_deriv, axis=1)
        
        return derivative_features
    
    def _calculate_areas(self, df):
        """Calculate various area-related features"""
        area_features = pd.DataFrame(index=df.index)
        
        # Total area under curve
        area_features['total_auc'] = np.trapz(df.values, axis=1)
        
        # Area ratios for different regions
        n_regions = 4
        region_size = df.shape[1] // n_regions
        for i in range(n_regions):
            start_idx = i * region_size
            end_idx = (i + 1) * region_size
            region_area = np.trapz(df.iloc[:, start_idx:end_idx].values, axis=1)
            area_features[f'region_{i+1}_area'] = region_area
            area_features[f'region_{i+1}_area_ratio'] = region_area / area_features['total_auc']
            
        return area_features
    
    def _analyze_frequency_domain(self, df):
        """Extract frequency domain features using FFT"""
        freq_features = pd.DataFrame(index=df.index)
        
        for idx, row in df.iterrows():
            # Compute FFT
            fft_vals = np.fft.fft(row.values)
            fft_freq = np.fft.fftfreq(len(row))
            
            # Extract features
            freq_features.at[idx, 'dominant_freq'] = abs(fft_freq[np.argmax(np.abs(fft_vals[1:]))])
            freq_features.at[idx, 'freq_magnitude'] = np.max(np.abs(fft_vals[1:]))
            freq_features.at[idx, 'freq_mean'] = np.mean(np.abs(fft_vals[1:]))
            freq_features.at[idx, '