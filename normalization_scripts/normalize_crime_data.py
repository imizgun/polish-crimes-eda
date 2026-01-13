"""
Skrypt do normalizacji danych o przestępczości z GUS
Konwertuje wide format (314 kolumn) do long format

STRUKTURA DANYCH:
- 26 kategorii przestępstw (+ wskaźniki wykrywalności + wskaźniki na 1000 mieszkańców)
- Każda kategoria ma 12 lat (2013-2024)
- 397 regionów (1 POLSKA + 16 województw + 380 powiatów)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import re


# Mapowanie kategorii do krótszych nazw (angielski)
CRIME_CATEGORIES_MAP = {
    'o charakterze kryminalnym': 'criminal',
    'o charakterze gospodarczym': 'economic',
    'przeciwko bezpieczeństwu powszechnemu i bezpieczeństwu w komunikacji - drogowe': 'traffic',
    'przeciwko życiu i zdrowiu': 'life_health',
    'przeciwko mieniu': 'property',
    'przeciwko wolności, wolności sumienia, wolności seksualnej i obyczajności razem': 'freedom_sexual',
    'przeciwko rodzinie i opiece': 'family',
    'przeciwko bezpieczeństwu powszechnemu i bezpieczeństwu w komunikacji razem': 'public_safety',
    
    # Wskaźniki wykrywalności (detection rates)
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - ogółem': 'detection_rate_total',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - o charakterze kryminalnym': 'detection_rate_criminal',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - o charakterze gospodarczym': 'detection_rate_economic',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - przeciwko bezpieczeństwu powszechnemu i bezpieczeństwu w komunikacji - drogowe': 'detection_rate_traffic',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - przeciwko życiu i zdrowiu': 'detection_rate_life_health',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - przeciwko mieniu': 'detection_rate_property',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - przeciwko wolności, wolności sumienia, wolności seksualnej i obyczajności': 'detection_rate_freedom_sexual',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - przeciwko rodzinie i opiece': 'detection_rate_family',
    'wskaźnik wykrywalności sprawców przestępstw stwierdzonych przez Policję - przeciwko bezpieczeństwu powszechnemu i bezpieczeństwu w komunikacji': 'detection_rate_public_safety',
    
    # Wskaźniki na 1000 mieszkańców (rates per 1000)
    'przestępstwa stwierdzone przez Policję ogółem na 1000 mieszkańców': 'rate_per_1000_total',
    'przestępstwa stwierdzone przez Policję o charakterze kryminalnym na 1000 mieszkańców': 'rate_per_1000_criminal',
    'przestępstwa stwierdzone przez Policję o charakterze gospodarczym na 1000 mieszkańców': 'rate_per_1000_economic',
    'przestępstwa stwierdzone przez Policję przeciwko bezpieczeństwu powszechnemu i bezpieczeństwu w komunikacji - drogowe na 1000 mieszkańców': 'rate_per_1000_traffic',
    'przestępstwa stwierdzone przez Policję przeciwko życiu i zdrowiu na 1000 mieszkańców': 'rate_per_1000_life_health',
    'przestępstwa stwierdzone przez Policję przeciwko mieniu na 1000 mieszkańców': 'rate_per_1000_property',
    'przestępstwa stwierdzone przez Policję przeciwko wolności, wolności sumienia i wyznania, wolności seksualnej i obyczajności na 1000 mieszkańców': 'rate_per_1000_freedom_sexual',
    'przestępstwa stwierdzone przez Policję przeciwko rodzinie i opiece na 1000 mieszkańców': 'rate_per_1000_family',
    'przestępstwa stwierdzone przez Policję przeciwko bezpieczeństwu powszecznemu i bezpieczeństwu w komunikacji na 1000 mieszkańców': 'rate_per_1000_public_safety',
}


def load_crime_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje surowe dane o przestępczości z pliku Excel
    
    Args:
        file_path: ścieżka do pliku Excel
    
    Returns:
        DataFrame z surowymi danymi (wide format)
    """
    # Wczytaj header (2 wiersze)
    df_header = pd.read_excel(file_path, sheet_name='TABLICA', nrows=2, header=None)
    
    # Przygotuj nazwy kolumn
    row0 = df_header.iloc[0].ffill()  # Kategoria przestępstwa
    row1 = df_header.iloc[1]          # Rok
    
    # Wczytaj dane (od 3 wiersza)
    df_data = pd.read_excel(file_path, sheet_name='TABLICA', skiprows=2)
    
    # Nadaj właściwe nazwy kolumnom
    new_columns = []
    for i in range(len(df_data.columns)):
        if i == 0:
            new_columns.append('region_code')
        elif i == 1:
            new_columns.append('region_name')
        else:
            category = str(row0.iloc[i])
            year = int(row1.iloc[i]) if pd.notna(row1.iloc[i]) else None
            
            # Użyj krótkiej nazwy kategorii
            category_short = CRIME_CATEGORIES_MAP.get(category, category)
            
            new_columns.append(f"{category_short}_{year}")
    
    df_data.columns = new_columns
    
    return df_data


def convert_to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje dane z wide format do long format
    
    Args:
        df_wide: DataFrame w wide format
    
    Returns:
        DataFrame w long format z kolumnami:
        - region_code: kod regionu
        - region_name: nazwa regionu
        - year: rok
        - crime_category: kategoria przestępstwa
        - value: wartość (liczba przestępstw, wskaźnik, etc.)
        - metric_type: typ metryki (count, detection_rate, rate_per_1000)
    """
    # Identyfikacja kolumn
    id_cols = ['region_code', 'region_name']
    value_cols = [col for col in df_wide.columns if col not in id_cols]
    
    # Melt
    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='variable',
        value_name='value'
    )
    
    # Parsowanie nazwy zmiennej (category_year)
    df_long[['crime_category', 'year']] = df_long['variable'].str.rsplit('_', n=1, expand=True)
    
    # Konwersja typów
    df_long['year'] = df_long['year'].astype(int)
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long['region_code'] = pd.to_numeric(df_long['region_code'], errors='coerce')
    
    # Określ typ metryki
    def get_metric_type(category):
        if category.startswith('detection_rate_'):
            return 'detection_rate'
        elif category.startswith('rate_per_1000_'):
            return 'rate_per_1000'
        else:
            return 'count'
    
    df_long['metric_type'] = df_long['crime_category'].apply(get_metric_type)
    
    # Usuń prefix z kategorii dla detection_rate i rate_per_1000
    def clean_category(row):
        cat = row['crime_category']
        if cat.startswith('detection_rate_'):
            return cat.replace('detection_rate_', '')
        elif cat.startswith('rate_per_1000_'):
            return cat.replace('rate_per_1000_', '')
        return cat
    
    df_long['crime_category'] = df_long.apply(clean_category, axis=1)
    
    # Usuń pomocniczą kolumnę
    df_long = df_long.drop('variable', axis=1)
    
    # Zmień kolejność kolumn
    df_long = df_long[['region_code', 'region_name', 'year', 'crime_category', 'metric_type', 'value']]
    
    # Sortuj
    df_long = df_long.sort_values(['region_code', 'year', 'crime_category', 'metric_type']).reset_index(drop=True)
    
    return df_long


def filter_powiaty_only(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Filtruje dane - zostawia tylko powiaty (bez POLSKA i województw)
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        DataFrame tylko z powiatami (region_code >= 200000)
    """
    # Kody powiatów zaczynają się od 200000
    df_powiaty = df_long[df_long['region_code'] >= 200000].copy()
    
    return df_powiaty


def pivot_by_metric_type(df_long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Tworzy osobne tabele dla każdego typu metryki
    
    Args:
        df_long: DataFrame w long format
    
    Returns:
        Słownik z 3 DataFrame:
        - 'count': liczba przestępstw
        - 'detection_rate': wskaźnik wykrywalności (%)
        - 'rate_per_1000': przestępstwa na 1000 mieszkańców
    """
    results = {}
    
    for metric_type in ['count', 'detection_rate', 'rate_per_1000']:
        df_metric = df_long[df_long['metric_type'] == metric_type].copy()
        df_metric = df_metric.drop('metric_type', axis=1)
        results[metric_type] = df_metric
    
    return results


def create_aggregated_counts(df_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy zagregowaną tabelę z całkowitą liczbą przestępstw
    (suma wszystkich kategorii)
    
    Args:
        df_counts: DataFrame z liczbami przestępstw (metric_type='count')
    
    Returns:
        DataFrame z kolumnami: region_code, region_name, year, total_crimes
    """
    # Filtruj główne kategorie (bez detection_rate i rate_per_1000)
    main_categories = ['criminal', 'economic', 'traffic', 'life_health', 
                      'property', 'freedom_sexual', 'family', 'public_safety']
    
    df_main = df_counts[df_counts['crime_category'].isin(main_categories)].copy()
    
    # Agreguj
    df_agg = df_main.groupby(['region_code', 'region_name', 'year'], as_index=False).agg({
        'value': 'sum'
    })
    
    df_agg.rename(columns={'value': 'total_crimes'}, inplace=True)
    
    return df_agg


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
        'unique_regions': df_long['region_code'].nunique(),
        'unique_powiaty': len(df_long[df_long['region_code'] >= 200000]['region_code'].unique()),
        'years_range': (df_long['year'].min(), df_long['year'].max()),
        'crime_categories': df_long['crime_category'].nunique(),
        'metric_types': df_long['metric_type'].unique().tolist(),
        'missing_values': df_long['value'].isna().sum(),
        'negative_values': (df_long['value'] < 0).sum()
    }
    
    return results


print("Wczytywanie danych o przestępczości...")
df_wide = load_crime_data('./data/przestepstwa_2013-2024.xlsx')
print(f"Rozmiar (wide format): {df_wide.shape}")
print(f"\nPierwsze wiersze:")
print(df_wide.head())

# 2. Konwertuj do long format
print("\n" + "="*80)
print("Konwersja do long format...")
df_long = convert_to_long_format(df_wide)
print(f"Rozmiar (long format): {df_long.shape}")
print(f"\nPierwsze wiersze:")
print(df_long.head(20))

# 3. Walidacja
print("\n" + "="*80)
print("Walidacja danych:")
validation = validate_data(df_long)
for key, value in validation.items():
    print(f"  {key}: {value}")

# 4. Filtruj tylko powiaty
print("\n" + "="*80)
print("Filtrowanie tylko powiatów...")
df_powiaty = filter_powiaty_only(df_long)
print(f"Rozmiar (tylko powiaty): {df_powiaty.shape}")
print(f"Liczba powiatów: {df_powiaty['region_code'].nunique()}")

# 5. Pivot według typu metryki
print("\n" + "="*80)
print("Podział według typu metryki...")
df_by_metric = pivot_by_metric_type(df_powiaty)

for metric_type, df in df_by_metric.items():
    print(f"\n{metric_type}: {df.shape[0]} wierszy")
    print(df.head())

# 6. Zagregowana tabela z całkowitą liczbą przestępstw
print("\n" + "="*80)
print("Tworzenie zagregowanej tabeli...")
df_total = create_aggregated_counts(df_by_metric['count'])
print(f"Rozmiar: {df_total.shape}")
print(df_total.head(15))

# Statystyki
print(f"\nStatystyki całkowitej liczby przestępstw:")
print(f"  Średnia: {df_total['total_crimes'].mean():.0f}")
print(f"  Mediana: {df_total['total_crimes'].median():.0f}")
print(f"  Min: {df_total['total_crimes'].min():.0f}")
print(f"  Max: {df_total['total_crimes'].max():.0f}")

# 7. Zapisz do plików
print("\n" + "="*80)
print("Zapisywanie do plików CSV...")

# Pełne dane - wszystkie regiony
df_long.to_csv('./output/crime/crime_long_format_all_regions.csv', index=False)

# Tylko powiaty
df_powiaty.to_csv('./output/crime/crime_long_format_powiaty.csv', index=False)

# Według typu metryki (tylko powiaty)
df_by_metric['count'].to_csv('./output/crime/crime_counts_powiaty.csv', index=False)
df_by_metric['detection_rate'].to_csv('./output/crime/crime_detection_rates_powiaty.csv', index=False)
df_by_metric['rate_per_1000'].to_csv('./output/crime/crime_rates_per_1000_powiaty.csv', index=False)

# Zagregowana tabela
df_total.to_csv('./output/crime/crime_total_powiaty.csv', index=False)

print("\n✓ Gotowe!")
print("\nUzyskane pliki:")
print("  - crime_long_format_all_regions.csv (wszystkie regiony)")
print("  - crime_long_format_powiaty.csv (tylko powiaty)")
print("  - crime_counts_powiaty.csv (liczby przestępstw)")
print("  - crime_detection_rates_powiaty.csv (wskaźniki wykrywalności)")
print("  - crime_rates_per_1000_powiaty.csv (wskaźniki na 1000 mieszkańców)")
print("  - crime_total_powiaty.csv (całkowita liczba przestępstw)")