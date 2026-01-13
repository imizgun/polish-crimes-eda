"""
Skrypt do normalizacji danych społeczno-ekonomicznych z GUS:
- Stopa bezrobocia (unemployment rate)
- Wynagrodzenie (wages - % of national average)
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_unemployment_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje dane o stopie bezrobocia
    
    Args:
        file_path: ścieżka do pliku Excel
    
    Returns:
        DataFrame z danymi w wide format
    """
    # Wczytaj header (2 wiersze)
    df_header = pd.read_excel(file_path, sheet_name='TABLICA', nrows=2, header=None)
    
    # Wczytaj dane (od 3 wiersza)
    df_data = pd.read_excel(file_path, sheet_name='TABLICA', skiprows=2)
    
    # Przygotuj nazwy kolumn
    # Lata są w wierszu 1 (indeks 1), kolumny 2-13
    new_columns = ['region_code', 'region_name']
    
    years = df_header.iloc[1, 2:].values  # Wiersz 1, od kolumny 2
    for year in years:
        if pd.notna(year):
            new_columns.append(f'unemployment_rate_{int(year)}')
    
    df_data.columns = new_columns
    
    return df_data


def load_wages_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje dane o wynagrodzeniach (% średniej krajowej)
    
    Args:
        file_path: ścieżka do pliku Excel
    
    Returns:
        DataFrame z danymi w wide format
    """
    # Wczytaj header (2 wiersze)
    df_header = pd.read_excel(file_path, sheet_name='TABLICA', nrows=2, header=None)
    
    # Wczytaj dane (od 3 wiersza)
    df_data = pd.read_excel(file_path, sheet_name='TABLICA', skiprows=2)
    
    # Przygotuj nazwy kolumn
    new_columns = ['region_code', 'region_name']
    
    # Lata są w wierszu 1 (indeks 1), od kolumny 2
    years = df_header.iloc[1, 2:].values
    for year in years:
        if pd.notna(year):
            new_columns.append(f'wage_index_{int(year)}')
    
    df_data.columns = new_columns
    
    return df_data


def convert_to_long_format(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Konwertuje dane z wide format do long format
    
    Args:
        df_wide: DataFrame w wide format
        value_name: nazwa kolumny wartości ('unemployment_rate' lub 'wage_index')
    
    Returns:
        DataFrame w long format
    """
    # Identyfikacja kolumn
    id_cols = ['region_code', 'region_name']
    value_cols = [col for col in df_wide.columns if col not in id_cols]
    
    # Melt
    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='year_col',
        value_name=value_name
    )
    
    # Wyciągnij rok z nazwy kolumny
    df_long['year'] = df_long['year_col'].str.extract(r'(\d{4})').astype(int)
    
    # Usuń pomocniczą kolumnę
    df_long = df_long.drop('year_col', axis=1)
    
    # Konwersja typów
    df_long['region_code'] = pd.to_numeric(df_long['region_code'], errors='coerce')
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    
    # Zmień kolejność kolumn
    df_long = df_long[['region_code', 'region_name', 'year', value_name]]
    
    # Sortuj
    df_long = df_long.sort_values(['region_code', 'year']).reset_index(drop=True)
    
    return df_long


def filter_powiaty_only(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Filtruje dane - zostawia tylko powiaty (bez POLSKA i województw)
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        DataFrame tylko z powiatami (region_code >= 200000)
    """
    df_powiaty = df_long[df_long['region_code'] >= 200000].copy()
    
    # Zmień nazwę kolumny
    df_powiaty = df_powiaty.rename(columns={'region_code': 'powiat_code', 'region_name': 'powiat_name'})
    
    return df_powiaty


def validate_data(df_long: pd.DataFrame, data_type: str) -> dict:
    """
    Waliduje jakość danych
    
    Args:
        df_long: DataFrame w long format
        data_type: 'unemployment' lub 'wages'
    
    Returns:
        Słownik z wynikami walidacji
    """
    value_col = 'unemployment_rate' if data_type == 'unemployment' else 'wage_index'
    
    results = {
        'total_rows': len(df_long),
        'unique_regions': df_long['region_code'].nunique() if 'region_code' in df_long.columns else df_long['powiat_code'].nunique(),
        'years_range': (df_long['year'].min(), df_long['year'].max()),
        'missing_values': df_long[value_col].isna().sum(),
        'negative_values': (df_long[value_col] < 0).sum() if df_long[value_col].notna().any() else 0,
        'mean': df_long[value_col].mean() if df_long[value_col].notna().any() else None,
        'median': df_long[value_col].median() if df_long[value_col].notna().any() else None,
        'min': df_long[value_col].min() if df_long[value_col].notna().any() else None,
        'max': df_long[value_col].max() if df_long[value_col].notna().any() else None
    }
    
    return results


def merge_socioeconomic_data(
    unemployment_path: str,
    wages_path: str
) -> pd.DataFrame:
    """
    Łączy dane o bezrobociu i wynagrodzeniach w jeden dataset
    
    Args:
        unemployment_path: ścieżka do unemployment_powiaty.csv
        wages_path: ścieżka do wages_powiaty.csv
    
    Returns:
        DataFrame z połączonymi danymi
    """
    df_unemp = pd.read_csv(unemployment_path)
    df_wages = pd.read_csv(wages_path)
    
    # Połącz
    df_merged = df_unemp.merge(
        df_wages[['powiat_code', 'year', 'wage_index']],
        on=['powiat_code', 'year'],
        how='outer'
    )
    
    return df_merged

print("="*80)
print("NORMALIZACJA DANYCH SPOŁECZNO-EKONOMICZNYCH")
print("="*80)

# ========================================================================
# 1. STOPA BEZROBOCIA
# ========================================================================

print("\n1. STOPA BEZROBOCIA")
print("-"*80)

# Wczytaj
print("Wczytywanie danych o bezrobociu...")
df_unemp_wide = load_unemployment_data('./data/stopa_bezrobocia_2013-2024.xlsx')
print(f"Rozmiar (wide format): {df_unemp_wide.shape}")

# Konwertuj do long format
print("Konwersja do long format...")
df_unemp_long = convert_to_long_format(df_unemp_wide, 'unemployment_rate')
print(f"Rozmiar (long format): {df_unemp_long.shape}")
print(f"\nPierwsze wiersze:")
print(df_unemp_long.head(10))

# Walidacja
print("\nWalidacja:")
validation = validate_data(df_unemp_long, 'unemployment')
for key, value in validation.items():
    print(f"  {key}: {value}")

# Filtruj tylko powiaty
print("\nFiltrowanie powiatów...")
df_unemp_powiaty = filter_powiaty_only(df_unemp_long)
print(f"Liczba powiatów: {df_unemp_powiaty['powiat_code'].nunique()}")

# Zapisz
df_unemp_long.to_csv('./output/socio/unemployment_all_regions.csv', index=False)
df_unemp_powiaty.to_csv('./output/socio/unemployment_powiaty.csv', index=False)

# ========================================================================
# 2. WYNAGRODZENIA
# ========================================================================

print("\n" + "="*80)
print("2. WYNAGRODZENIA")
print("-"*80)

# Wczytaj
print("Wczytywanie danych o wynagrodzeniach...")
df_wages_wide = load_wages_data('./data/wynagrodzenie_2013-2024.xlsx')
print(f"Rozmiar (wide format): {df_wages_wide.shape}")

# Konwertuj do long format
print("Konwersja do long format...")
df_wages_long = convert_to_long_format(df_wages_wide, 'wage_index')
print(f"Rozmiar (long format): {df_wages_long.shape}")
print(f"\nPierwsze wiersze:")
print(df_wages_long.head(10))

# Walidacja
print("\nWalidacja:")
validation = validate_data(df_wages_long, 'wages')
for key, value in validation.items():
    print(f"  {key}: {value}")

# Filtruj tylko powiaty
print("\nFiltrowanie powiatów...")
df_wages_powiaty = filter_powiaty_only(df_wages_long)
print(f"Liczba powiatów: {df_wages_powiaty['powiat_code'].nunique()}")

# Zapisz
df_wages_long.to_csv('./output/socio/wages_all_regions.csv', index=False)
df_wages_powiaty.to_csv('./output/socio/wages_powiaty.csv', index=False)

# ========================================================================
# 3. POŁĄCZONE DANE SPOŁECZNO-EKONOMICZNE
# ========================================================================

print("\n" + "="*80)
print("3. ŁĄCZENIE DANYCH")
print("-"*80)

df_socioeconomic = merge_socioeconomic_data(
    './output/socio/unemployment_powiaty.csv',
    './output/socio/wages_powiaty.csv'
)

print(f"Rozmiar połączonego datasetu: {df_socioeconomic.shape}")
print(f"\nPierwsze wiersze:")
print(df_socioeconomic.head(15))

print(f"\nStatystyki:")
print(f"  Liczba powiatów: {df_socioeconomic['powiat_code'].nunique()}")
print(f"  Lata: {df_socioeconomic['year'].min()} - {df_socioeconomic['year'].max()}")
print(f"  Braki unemployment_rate: {df_socioeconomic['unemployment_rate'].isna().sum()}")
print(f"  Braki wage_index: {df_socioeconomic['wage_index'].isna().sum()}")

# Zapisz
df_socioeconomic.to_csv('./output/socio/economic_powiaty.csv', index=False)

# ========================================================================
# PODSUMOWANIE
# ========================================================================

print("\n" + "="*80)
print("✓ GOTOWE!")
print("="*80)
print("\nUzyskane pliki:")
print("  Bezrobocie:")
print("    - unemployment_all_regions.csv (wszystkie regiony)")
print("    - unemployment_powiaty.csv (tylko powiaty)")
print("  Wynagrodzenia:")
print("    - wages_all_regions.csv (wszystkie regiony)")
print("    - wages_powiaty.csv (tylko powiaty)")
print("  Połączone:")
print("    - socioeconomic_powiaty.csv (bezrobocie + wynagrodzenia)")

print("\n" + "="*80)
print("KLUCZOWE STATYSTYKI")
print("="*80)

print(f"\nBezrobocie (powiaty):")
print(f"  Średnia: {df_unemp_powiaty['unemployment_rate'].mean():.2f}%")
print(f"  Mediana: {df_unemp_powiaty['unemployment_rate'].median():.2f}%")
print(f"  Zakres: {df_unemp_powiaty['unemployment_rate'].min():.2f}% - {df_unemp_powiaty['unemployment_rate'].max():.2f}%")

print(f"\nWynagrodzenia (% średniej krajowej):")
print(f"  Średnia: {df_wages_powiaty['wage_index'].mean():.2f}%")
print(f"  Mediana: {df_wages_powiaty['wage_index'].median():.2f}%")
print(f"  Zakres: {df_wages_powiaty['wage_index'].min():.2f}% - {df_wages_powiaty['wage_index'].max():.2f}%")