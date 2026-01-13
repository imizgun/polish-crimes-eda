"""
FINALNA INTEGRACJA WSZYSTKICH DANYCH
Łączy:
- Przestępstwa
- Demografia (ludność, wiek, płeć, gęstość)
- Bezrobocie
- Wynagrodzenia
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Ścieżka do katalogu głównego projektu
PROJECT_ROOT = Path(__file__).parent.parent


def create_complete_dataset() -> pd.DataFrame:
    """
    Tworzy kompletny dataset ze WSZYSTKIMI danymi
    
    Returns:
        DataFrame z pełnymi danymi
    """
    print("Wczytywanie danych...")

    # 1. Załaduj poprzedni zintegrowany dataset (przestępstwa + demografia) - KODY JAKO STRING!
    df_base = pd.read_csv(PROJECT_ROOT / 'output' / 'integrated_crime_data.csv', dtype={'powiat_code': str})
    print(f"  Dataset bazowy: {df_base.shape}")

    # 2. Załaduj dane społeczno-ekonomiczne - KODY JAKO STRING!
    df_socio = pd.read_csv(PROJECT_ROOT / 'output' / 'socio' / 'economic_powiaty.csv', dtype={'powiat_code': str})
    print(f"  Dane społeczno-ekonomiczne: {df_socio.shape}")
    
    # 3. Połącz
    print("\nŁączenie danych...")
    df_complete = df_base.merge(
        df_socio[['powiat_code', 'year', 'unemployment_rate', 'wage_index']],
        on=['powiat_code', 'year'],
        how='left'
    )
    
    print(f"  Rozmiar po merge: {df_complete.shape}")
    
    return df_complete


def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje dodatkowe cechy pochodne
    
    Args:
        df: DataFrame z danymi
    
    Returns:
        DataFrame z dodatkowymi cechami
    """
    df_enhanced = df.copy()
    
    # Kategoria bezrobocia
    df_enhanced['unemployment_category'] = pd.cut(
        df_enhanced['unemployment_rate'],
        bins=[0, 5, 10, 15, 100],
        labels=['very_low', 'low', 'medium', 'high']
    )
    
    # Kategoria wynagrodzeń (względem średniej krajowej)
    df_enhanced['wage_category'] = pd.cut(
        df_enhanced['wage_index'],
        bins=[0, 80, 90, 100, 200],
        labels=['low', 'medium_low', 'medium_high', 'high']
    )
    
    # Wskaźnik złożony: bezrobocie wysokie + płace niskie = ryzyko społeczne
    df_enhanced['social_risk_score'] = (
        (df_enhanced['unemployment_rate'] - df_enhanced['unemployment_rate'].mean()) / 
        df_enhanced['unemployment_rate'].std() -
        (df_enhanced['wage_index'] - df_enhanced['wage_index'].mean()) / 
        df_enhanced['wage_index'].std()
    )
    
    return df_enhanced


def validate_complete_dataset(df: pd.DataFrame) -> dict:
    """
    Waliduje kompletny dataset
    
    Args:
        df: kompletny DataFrame
    
    Returns:
        Słownik z wynikami walidacji
    """
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'unique_powiaty': df['powiat_code'].nunique(),
        'years_range': (df['year'].min(), df['year'].max()),
        'missing_crime': df['total_crimes'].isna().sum(),
        'missing_population': df['total_population'].isna().sum(),
        'missing_unemployment': df['unemployment_rate'].isna().sum(),
        'missing_wages': df['wage_index'].isna().sum(),
        'complete_rows': len(df.dropna())
    }
    
    return results


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza korelacje między kluczowymi zmiennymi
    
    Args:
        df: DataFrame z danymi
    
    Returns:
        DataFrame z korelacjami
    """
    # Wybierz kluczowe cechy numeryczne
    key_features = [
        'crime_rate_per_100k',
        'population_density',
        'youth_ratio',
        'gender_ratio',
        'unemployment_rate',
        'wage_index',
        'population_change_pct'
    ]
    
    # Usuń brakujące
    df_clean = df[key_features].dropna()
    
    # Oblicz korelacje
    corr_matrix = df_clean.corr()
    
    # Wyciągnij korelacje z crime_rate_per_100k
    crime_corr = corr_matrix['crime_rate_per_100k'].sort_values(ascending=False)
    
    return crime_corr

print("="*80)
print("FINALNA INTEGRACJA WSZYSTKICH DANYCH")
print("="*80)
print()

# 1. Utwórz kompletny dataset
df_complete = create_complete_dataset()

# 2. Dodaj dodatkowe cechy
print("\nDodawanie cech pochodnych...")
df_enhanced = add_additional_features(df_complete)
print(f"  Liczba kolumn: {len(df_enhanced.columns)}")

# 3. Walidacja
print("\n" + "="*80)
print("WALIDACJA DATASETU")
print("="*80)
validation = validate_complete_dataset(df_enhanced)
for key, value in validation.items():
    print(f"  {key}: {value}")

# 4. Oblicz korelacje
print("\n" + "="*80)
print("KORELACJE Z WSKAŹNIKIEM PRZESTĘPCZOŚCI")
print("="*80)
correlations = calculate_correlations(df_enhanced)
print(correlations)

# 5. Statystyki opisowe
print("\n" + "="*80)
print("STATYSTYKI OPISOWE")
print("="*80)

key_vars = ['crime_rate_per_100k', 'population_density', 'unemployment_rate', 
            'wage_index', 'youth_ratio']

print("\nPodstawowe statystyki:")
print(df_enhanced[key_vars].describe())

# 6. Zapisz dane
print("\n" + "="*80)
print("ZAPISYWANIE PLIKÓW")
print("="*80)

# Pełny dataset
df_enhanced.to_csv(PROJECT_ROOT / 'output' / 'complete_dataset.csv', index=False)
print("  ✓ complete_dataset.csv")

# Tylko 2024 rok
df_2024 = df_enhanced[df_enhanced['year'] == 2024]
df_2024.to_csv(PROJECT_ROOT / 'output' / 'complete_dataset_2024.csv', index=False)
print("  ✓ complete_dataset_2024.csv")

# Dataset dla modelu ML (bez braków)
df_ml = df_enhanced.dropna(subset=[
    'crime_rate_per_100k', 'population_density', 'youth_ratio',
    'unemployment_rate', 'wage_index', 'gender_ratio', 'population_change_pct'
])
df_ml.to_csv(PROJECT_ROOT / 'output' / 'ml_ready_dataset.csv', index=False)
print(f"  ✓ ml_ready_dataset.csv ({len(df_ml)} wierszy - bez braków)")

# 7. Analiza kategorii
print("\n" + "="*80)
print("ANALIZA WEDŁUG KATEGORII (2024)")
print("="*80)

if len(df_2024) > 0:
    print("\nPrzestępczość według kategorii urbanizacji:")
    if 'urbanization_category' in df_2024.columns:
        urban_stats = df_2024.groupby('urbanization_category')['crime_rate_per_100k'].agg(['mean', 'median', 'count'])
        print(urban_stats)
    
    print("\nPrzestępczość według kategorii bezrobocia:")
    if 'unemployment_category' in df_2024.columns:
        unemp_stats = df_2024.groupby('unemployment_category')['crime_rate_per_100k'].agg(['mean', 'median', 'count'])
        print(unemp_stats)
    
    print("\nPrzestępczość według kategorii wynagrodzeń:")
    if 'wage_category' in df_2024.columns:
        wage_stats = df_2024.groupby('wage_category')['crime_rate_per_100k'].agg(['mean', 'median', 'count'])
        print(wage_stats)

# 8. Podsumowanie
print("\n" + "="*80)
print("✓ KOMPLETNY DATASET GOTOWY!")
print("="*80)
print(f"\nRozmiar: {df_enhanced.shape[0]} wierszy × {df_enhanced.shape[1]} kolumn")
print(f"Powiaty: {df_enhanced['powiat_code'].nunique()}")
print(f"Lata: {df_enhanced['year'].min()} - {df_enhanced['year'].max()}")
print(f"\nKompletne obserwacje (bez braków): {validation['complete_rows']}")
print(f"Dataset ML (wybrane cechy bez braków): {len(df_ml)} wierszy")

print("\nUzyskane pliki:")
print("  1. complete_dataset.csv - PEŁNY dataset (wszystkie dane)")
print("  2. complete_dataset_2024.csv - tylko 2024 rok")
print("  3. ml_ready_dataset.csv - gotowy do ML (bez braków)")

print("\n" + "="*80)
print("CECHY DOSTĘPNE DO MODELU ML:")
print("="*80)
print("\nCechy demograficzne:")
print("  - population_density (gęstość zaludnienia)")
print("  - youth_ratio (udział młodzieży)")
print("  - middle_age_ratio (udział osób w średnim wieku)")
print("  - elderly_ratio (udział osób starszych)")
print("  - gender_ratio (wskaźnik płci)")
print("  - population_change_pct (zmiana populacji)")

print("\nCechy społeczno-ekonomiczne:")
print("  - unemployment_rate (stopa bezrobocia)")
print("  - wage_index (wynagrodzenia % średniej)")

print("\nCechy kategoryczne:")
print("  - urbanization_category (rural/suburban/urban/metro)")
print("  - unemployment_category (very_low/low/medium/high)")
print("  - wage_category (low/medium_low/medium_high/high)")
print("  - population_size_category (small/medium/large/very_large)")