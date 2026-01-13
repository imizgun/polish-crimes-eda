"""
Skrypt do normalizacji danych o powierzchni powiatów z GUS
Konwertuje wide format (14 kolumn: Kod, Nazwa, 2013-2024) do long format
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_area_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje dane o powierzchni z pliku Excel
    
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
    new_columns = ['powiat_code', 'powiat_name']
    
    # Dodaj lata (kolumny 2-13 to lata 2013-2024)
    years = df_header.iloc[1, 2:].values
    for year in years:
        if pd.notna(year):
            new_columns.append(f'area_km2_{int(year)}')
    
    df_data.columns = new_columns
    
    return df_data


def convert_to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje dane z wide format do long format
    
    Args:
        df_wide: DataFrame w wide format
    
    Returns:
        DataFrame w long format z kolumnami:
        - powiat_code: kod powiatu
        - powiat_name: nazwa powiatu
        - year: rok
        - area_km2: powierzchnia w km²
    """
    # Identyfikacja kolumn
    id_cols = ['powiat_code', 'powiat_name']
    value_cols = [col for col in df_wide.columns if col.startswith('area_km2_')]
    
    # Melt
    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='year_col',
        value_name='area_km2'
    )
    
    # Wyciągnij rok z nazwy kolumny (area_km2_2013 -> 2013)
    df_long['year'] = df_long['year_col'].str.extract(r'(\d{4})').astype(int)
    
    # Usuń pomocniczą kolumnę
    df_long = df_long.drop('year_col', axis=1)
    
    # Konwersja typów
    df_long['powiat_code'] = df_long['powiat_code'].astype(int)
    df_long['area_km2'] = pd.to_numeric(df_long['area_km2'], errors='coerce')
    
    # Zmień kolejność kolumn
    df_long = df_long[['powiat_code', 'powiat_name', 'year', 'area_km2']]
    
    # Sortuj
    df_long = df_long.sort_values(['powiat_code', 'year']).reset_index(drop=True)
    
    return df_long


def calculate_area_statistics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza statystyki powierzchni (sprawdza stabilność danych)
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        DataFrame ze statystykami dla każdego powiatu
    """
    stats = df_long.groupby('powiat_code').agg({
        'powiat_name': 'first',
        'area_km2': ['min', 'max', 'mean', 'std']
    }).reset_index()
    
    stats.columns = ['powiat_code', 'powiat_name', 'area_min', 'area_max', 'area_mean', 'area_std']
    
    # Sprawdź, czy są zmiany w powierzchni (powinno być stałe)
    stats['area_changed'] = stats['area_min'] != stats['area_max']
    
    return stats


def get_latest_area(df_long: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
    """
    Pobiera dane o powierzchni dla najnowszego roku (2024)
    Użycie: do łączenia z innymi datasetem
    
    Args:
        df_long: DataFrame w long format
        year: rok do pobrania (domyślnie 2024)
    
    Returns:
        DataFrame z kolumnami: powiat_code, powiat_name, area_km2
    """
    df_latest = df_long[df_long['year'] == year].copy()
    df_latest = df_latest.drop('year', axis=1)
    
    return df_latest


def validate_data(df_long: pd.DataFrame) -> dict:
    """
    Waliduje jakość danych
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        Słownik z wynikami walidacji
    """
    results = {
        'total_rows': len(df_long),
        'unique_powiaty': df_long['powiat_code'].nunique(),
        'years_range': (df_long['year'].min(), df_long['year'].max()),
        'missing_values': df_long['area_km2'].isna().sum(),
        'negative_values': (df_long['area_km2'] < 0).sum(),
        'zero_values': (df_long['area_km2'] == 0).sum(),
        'mean_area_km2': df_long['area_km2'].mean(),
        'median_area_km2': df_long['area_km2'].median()
    }
    
    # Sprawdź stabilność (powierzchnia powinna być stała w czasie)
    area_changes = df_long.groupby('powiat_code')['area_km2'].nunique()
    results['powiaty_with_area_changes'] = (area_changes > 1).sum()
    
    return results


def merge_with_population(df_area: pd.DataFrame, df_population: pd.DataFrame) -> pd.DataFrame:
    """
    Łączy dane o powierzchni z danymi o ludności
    Oblicza gęstość zaludnienia
    
    Args:
        df_area: DataFrame z danymi o powierzchni (long format)
        df_population: DataFrame z danymi o ludności (total population)
    
    Returns:
        DataFrame z dodaną kolumną population_density (osoby/km²)
    """
    df_merged = df_population.merge(
        df_area,
        on=['powiat_code', 'powiat_name', 'year'],
        how='left'
    )
    
    # Oblicz gęstość zaludnienia (osoby na km²)
    df_merged['population_density'] = (
        df_merged['total_population'] / df_merged['area_km2']
    ).round(2)
    
    return df_merged

# 1. Wczytaj dane o powierzchni
print("Wczytywanie danych o powierzchni...")
df_wide = load_area_data('./data/powierzchnia_2013-2024.xlsx')
print(f"Rozmiar (wide format): {df_wide.shape}")
print(f"\nPierwsze wiersze:")
print(df_wide.head())

# 2. Konwertuj do long format
print("\n" + "="*80)
print("Konwersja do long format...")
df_long = convert_to_long_format(df_wide)
print(f"Rozmiar (long format): {df_long.shape}")
print(f"\nPierwsze wiersze:")
print(df_long.head(15))

# 3. Walidacja
print("\n" + "="*80)
print("Walidacja danych:")
validation = validate_data(df_long)
for key, value in validation.items():
    print(f"  {key}: {value}")

# 4. Statystyki powierzchni
print("\n" + "="*80)
print("Statystyki powierzchni (sprawdzenie stabilności):")
stats = calculate_area_statistics(df_long)
print(stats.head(10))

# Ile powiatów ma zmiany w powierzchni?
changed = stats[stats['area_changed']]
if len(changed) > 0:
    print(f"\nUwaga! {len(changed)} powiatów ma zmiany w powierzchni:")
    print(changed[['powiat_name', 'area_min', 'area_max']])
else:
    print("\n✓ Powierzchnia jest stabilna dla wszystkich powiatów")

# 5. Pobierz dane dla 2024
print("\n" + "="*80)
print("Dane dla roku 2024:")
df_2024 = get_latest_area(df_long, year=2024)
print(df_2024.head())
print(f"\nŚrednia powierzchnia powiatu: {df_2024['area_km2'].mean():.2f} km²")
print(f"Min: {df_2024['area_km2'].min():.2f} km², Max: {df_2024['area_km2'].max():.2f} km²")

# 6. Połącz z danymi o ludności (jeśli istnieją)
print("\n" + "="*80)
print("Łączenie z danymi o ludności...")
try:
    df_population = pd.read_csv('./output/area/population_total.csv')
    df_with_density = merge_with_population(df_long, df_population)
    
    print(f"Rozmiar po merge: {df_with_density.shape}")
    print(f"\nPierwsze wiersze:")
    print(df_with_density.head(10))
    
    print(f"\nStatystyki gęstości zaludnienia:")
    print(f"  Średnia: {df_with_density['population_density'].mean():.2f} osób/km²")
    print(f"  Mediana: {df_with_density['population_density'].median():.2f} osób/km²")
    print(f"  Min: {df_with_density['population_density'].min():.2f} osób/km²")
    print(f"  Max: {df_with_density['population_density'].max():.2f} osób/km²")
    
    # Zapisz połączone dane
    df_with_density.to_csv('./output/area/population_with_density.csv', index=False)
    print("\n✓ Zapisano: population_with_density.csv")
    
except FileNotFoundError:
    print("Plik population_total.csv nie znaleziony. Pomiń merge.")

# 7. Zapisz do plików
print("\n" + "="*80)
print("Zapisywanie do plików CSV...")
df_long.to_csv('./output/area/area_long_format.csv', index=False)
df_2024.to_csv('./output/area/area_2024.csv', index=False)
stats.to_csv('./output/area/area_statistics.csv', index=False)

print("\n✓ Gotowe!")
print("\nUzyskane pliki:")
print("  - area_long_format.csv (pełne dane 2013-2024)")
print("  - area_2024.csv (tylko najnowszy rok)")
print("  - area_statistics.csv (statystyki)")
print("  - population_with_density.csv (ludność + gęstość zaludnienia)")