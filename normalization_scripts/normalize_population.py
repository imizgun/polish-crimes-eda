"""
Skrypt do normalizacji danych o ludności z GUS
Konwertuje wide format (290 kolumn) do long format (tidy data)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def load_population_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje surowe dane o ludności z pliku Excel
    
    Args:
        file_path: ścieżka do pliku Excel
    
    Returns:
        DataFrame z surowymi danymi (wide format)
    """
    # Najpierw wczytaj header (3 wiersze)
    df_header = pd.read_excel(file_path, sheet_name='TABLICA', nrows=3, header=None)
    
    # Przygotuj nazwy kolumn z multi-level header
    row0 = df_header.iloc[0].ffill()  # Grupa wiekowa
    row1 = df_header.iloc[1].ffill()  # Płeć
    row2 = df_header.iloc[2]          # Rok
    
    # Wczytaj dane właściwe (od 4 wiersza)
    df_data = pd.read_excel(file_path, sheet_name='TABLICA', skiprows=3)
    
    # Nadaj właściwe nazwy kolumnom
    new_columns = []
    for i in range(len(df_data.columns)):
        if i == 0:
            new_columns.append('powiat_code')
        elif i == 1:
            new_columns.append('powiat_name')
        else:
            age = str(row0.iloc[i]).replace('ogółem', 'total')
            gender = str(row1.iloc[i])
            year = int(row2.iloc[i]) if pd.notna(row2.iloc[i]) else None
            
            # Mapowanie płci na EN
            gender_map = {'mężczyźni': 'male', 'kobiety': 'female'}
            gender_en = gender_map.get(gender, gender)
            
            new_columns.append(f"{age}_{gender_en}_{year}")
    
    df_data.columns = new_columns
    
    return df_data


def convert_to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje dane z wide format do long format (tidy data)
    
    Args:
        df_wide: DataFrame w wide format (290 kolumn)
    
    Returns:
        DataFrame w long format z kolumnami:
        - powiat_code: kod powiatu
        - powiat_name: nazwa powiatu
        - year: rok
        - age_group: grupa wiekowa
        - gender: płeć (male/female)
        - population: liczba ludności
    """
    # Identyfikacja kolumn
    id_cols = ['powiat_code', 'powiat_name']
    value_cols = [col for col in df_wide.columns if col not in id_cols]
    
    # Melt - przekształcenie do long format
    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='variable',
        value_name='population'
    )
    
    # Parsowanie nazwy zmiennej (age_gender_year)
    df_long[['age_group', 'gender', 'year']] = df_long['variable'].str.rsplit('_', n=2, expand=True)
    
    # Konwersja typów
    df_long['year'] = df_long['year'].astype(int)
    df_long['population'] = pd.to_numeric(df_long['population'], errors='coerce')
    df_long['powiat_code'] = df_long['powiat_code'].astype(int)
    
    # Usuń pomocniczą kolumnę
    df_long = df_long.drop('variable', axis=1)
    
    # Zmień kolejność kolumn
    df_long = df_long[['powiat_code', 'powiat_name', 'year', 'age_group', 'gender', 'population']]
    
    # Sortuj
    df_long = df_long.sort_values(['powiat_code', 'year', 'age_group', 'gender']).reset_index(drop=True)
    
    return df_long


def create_total_population_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy zagregowaną tabelę z całkowitą populacją według powiatu i roku
    (suma wszystkich grup wiekowych i płci)
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        DataFrame z kolumnami: powiat_code, powiat_name, year, total_population
    """
    # Filtruj tylko 'total' (ogółem) - już zawiera sumę wszystkich grup
    df_total = df_long[df_long['age_group'] == 'total'].copy()
    
    # Grupuj po powiecie i roku, sumując po płci
    df_agg = df_total.groupby(['powiat_code', 'powiat_name', 'year'], as_index=False).agg({
        'population': 'sum'
    })
    
    df_agg.rename(columns={'population': 'total_population'}, inplace=True)
    
    return df_agg


def create_gender_aggregation(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy tabelę z podziałem na płeć (bez grup wiekowych)
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        DataFrame z kolumnami: powiat_code, powiat_name, year, 
                               male_population, female_population, gender_ratio
    """
    # Filtruj tylko 'total' age_group
    df_gender = df_long[df_long['age_group'] == 'total'].copy()
    
    # Pivot by gender
    df_pivot = df_gender.pivot_table(
        index=['powiat_code', 'powiat_name', 'year'],
        columns='gender',
        values='population',
        aggfunc='sum'
    ).reset_index()
    
    # Oblicz wskaźnik płci (męzczyzn na 100 kobiet)
    df_pivot['gender_ratio'] = (df_pivot['male'] / df_pivot['female'] * 100).round(2)
    
    df_pivot.rename(columns={
        'male': 'male_population',
        'female': 'female_population'
    }, inplace=True)
    
    return df_pivot


def create_age_aggregation(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy tabelę z grupami wiekowymi (bez podziału na płeć)
    
    Agreguje do szerokich grup:
    - age_15_29: młodzi (15-29 lat)
    - age_30_49: średni wiek (30-49 lat)  
    - age_50_69: starsi (50-69 lat)
    - age_total: wszystkie grupy
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        DataFrame z kolumnami: powiat_code, powiat_name, year, age_15_29, age_30_49, age_50_69
    """
    # Mapowanie grup wiekowych do szerokich kategorii
    age_mapping = {
        '15-19': 'age_15_29',
        '20-24': 'age_15_29',
        '25-29': 'age_15_29',
        '30-34': 'age_30_49',
        '35-39': 'age_30_49',
        '40-44': 'age_30_49',
        '45-49': 'age_30_49',
        '50-54': 'age_50_69',
        '55-59': 'age_50_69',
        '60-64': 'age_50_69',
        '65-69': 'age_50_69'
    }
    
    # Filtruj tylko szczegółowe grupy wiekowe (bez 'total')
    df_age = df_long[df_long['age_group'] != 'total'].copy()
    
    # Dodaj szeroką kategorię wiekową
    df_age['age_category'] = df_age['age_group'].map(age_mapping)
    
    # Usuń wiersze, które nie pasują do żadnej kategorii
    df_age = df_age.dropna(subset=['age_category'])
    
    # Agreguj po powiecie, roku i szerokiej kategorii (sumuj po płci i szczegółowych grupach)
    df_agg = df_age.groupby(['powiat_code', 'powiat_name', 'year', 'age_category'], as_index=False).agg({
        'population': 'sum'
    })
    
    # Pivot
    df_pivot = df_agg.pivot_table(
        index=['powiat_code', 'powiat_name', 'year'],
        columns='age_category',
        values='population',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    return df_pivot


def calculate_derived_features(df_total: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza pochodne cechy demograficzne
    
    Args:
        df_total: DataFrame z całkowitą populacją (powiat, rok)
    
    Returns:
        DataFrame z dodatkowymi kolumnami:
        - population_change: zmiana bezwzględna rok do roku
        - population_change_pct: zmiana procentowa rok do roku
    """
    df = df_total.copy()
    
    # Sortuj według powiatu i roku
    df = df.sort_values(['powiat_code', 'year'])
    
    # Oblicz zmianę rok do roku
    df['population_change'] = df.groupby('powiat_code')['total_population'].diff()
    
    # Oblicz procentową zmianę
    df['population_change_pct'] = (
        df.groupby('powiat_code')['total_population'].pct_change() * 100
    ).round(2)
    
    return df


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
        'missing_values': df_long['population'].isna().sum(),
        'negative_values': (df_long['population'] < 0).sum(),
        'zero_values': (df_long['population'] == 0).sum()
    }
    
    return results


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

print("Wczytywanie danych...")
df_wide = load_population_data('./data/ludnosc_2013-2024.xlsx')
print(f"Rozmiar (wide format): {df_wide.shape}")

# 2. Konwertuj do long format
print("\nKonwersja do long format...")
df_long = convert_to_long_format(df_wide)
print(f"Rozmiar (long format): {df_long.shape}")
print(f"\nPierwsze wiersze:")
print(df_long.head(10))

# 3. Walidacja
print("\nWalidacja danych:")
validation = validate_data(df_long)
for key, value in validation.items():
    print(f"  {key}: {value}")

# 4. Utwórz zagregowane tabele
print("\nTworzenie zagregowanych tabel...")

# 4a. Całkowita populacja
df_total = create_total_population_table(df_long)
print(f"Tabela całkowita: {df_total.shape}")
print(df_total.head())

# 4b. Podział na płeć
df_gender = create_gender_aggregation(df_long)
print(f"\nTabela płci: {df_gender.shape}")
print(df_gender.head())

# 4c. Grupy wiekowe
df_age = create_age_aggregation(df_long)
print(f"\nTabela wieku: {df_age.shape}")
print(df_age.head())

# 4d. Cechy pochodne
df_with_changes = calculate_derived_features(df_total)
print(f"\nTabela z cechami pochodnymi: {df_with_changes.shape}")
print(df_with_changes.head(15))

# 5. Zapisz do plików
print("\nZapisywanie do plików CSV...")
df_long.to_csv('./output/population/population_long_format.csv', index=False)
df_total.to_csv('./output/population/population_total.csv', index=False)
df_gender.to_csv('./output/population/population_gender.csv', index=False)
df_age.to_csv('./output/population/population_age_groups.csv', index=False)
df_with_changes.to_csv('./output/population/population_with_changes.csv', index=False)

print("\n✓ Gotowe!")
print("\nUzyskane pliki:")
print("  - population_long_format.csv (pełne dane)")
print("  - population_total.csv (suma po powiecie i roku)")
print("  - population_gender.csv (podział na płeć)")
print("  - population_age_groups.csv (grupy wiekowe)")
print("  - population_with_changes.csv (z cechami pochodnymi)")