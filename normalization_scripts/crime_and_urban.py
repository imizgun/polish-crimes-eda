"""
Skrypt integracji wszystkich danych:
- Przestępstwa
- Ludność (w tym gęstość zaludnienia)
- Cechy demograficzne (płeć, wiek)
- Obliczenie wskaźników przestępczości
"""

import pandas as pd
import numpy as np


def create_integrated_dataset(
    crime_path: str,
    population_density_path: str,
    population_gender_path: str,
    population_age_path: str,
    population_changes_path: str
) -> pd.DataFrame:
    """
    Tworzy zintegrowany dataset ze wszystkimi danymi
    
    Args:
        crime_path: ścieżka do crime_total_powiaty.csv
        population_density_path: ścieżka do population_with_density.csv
        population_gender_path: ścieżka do population_gender.csv
        population_age_path: ścieżka do population_age_groups.csv
        population_changes_path: ścieżka do population_with_changes.csv
    
    Returns:
        DataFrame z zintegrowanymi danymi
    """
    # 1. Wczytaj dane o przestępczości
    df_crime = pd.read_csv(crime_path)
    
    # Zmień nazwę kolumny region_code na powiat_code
    df_crime = df_crime.rename(columns={
        'region_code': 'powiat_code',
        'region_name': 'powiat_name'
    })
    
    # 2. Wczytaj dane demograficzne
    df_density = pd.read_csv(population_density_path)
    df_gender = pd.read_csv(population_gender_path)
    df_age = pd.read_csv(population_age_path)
    df_changes = pd.read_csv(population_changes_path)
    
    # 3. Połącz wszystko
    df = df_crime.copy()
    
    # JOIN z gęstością zaludnienia
    df = df.merge(
        df_density[['powiat_code', 'year', 'total_population', 'area_km2', 'population_density']],
        on=['powiat_code', 'year'],
        how='left'
    )
    
    # JOIN z danymi o płci
    df = df.merge(
        df_gender[['powiat_code', 'year', 'male_population', 'female_population', 'gender_ratio']],
        on=['powiat_code', 'year'],
        how='left'
    )
    
    # JOIN z grupami wiekowymi
    df = df.merge(
        df_age[['powiat_code', 'year', 'age_15_29', 'age_30_49', 'age_50_69']],
        on=['powiat_code', 'year'],
        how='left'
    )
    
    # JOIN z zmianami populacji
    df = df.merge(
        df_changes[['powiat_code', 'year', 'population_change', 'population_change_pct']],
        on=['powiat_code', 'year'],
        how='left'
    )
    
    # 4. Oblicz wskaźniki przestępczości
    
    # Wskaźnik na 100 tys. mieszkańców (podstawowy)
    df['crime_rate_per_100k'] = (df['total_crimes'] / df['total_population'] * 100000).round(2)
    
    # Wskaźnik na 1000 mieszkańców (dla porównania z danymi GUS)
    df['crime_rate_per_1000'] = (df['total_crimes'] / df['total_population'] * 1000).round(2)
    
    # 5. Feature engineering - cechy pochodne
    
    # Udział młodzieży w populacji
    df['youth_ratio'] = (df['age_15_29'] / df['total_population']).round(4)
    
    # Udział osób w wieku średnim
    df['middle_age_ratio'] = (df['age_30_49'] / df['total_population']).round(4)
    
    # Udział osób starszych
    df['elderly_ratio'] = (df['age_50_69'] / df['total_population']).round(4)
    
    # Kategoria urbanizacji
    df['urbanization_category'] = pd.cut(
        df['population_density'],
        bins=[0, 100, 300, 1000, 5000],
        labels=['rural', 'suburban', 'urban', 'metro']
    )
    
    # Kategoria wielkości powiatu (populacja)
    df['population_size_category'] = pd.cut(
        df['total_population'],
        bins=[0, 30000, 60000, 100000, 1000000],
        labels=['small', 'medium', 'large', 'very_large']
    )
    
    return df


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza dodatkowe statystyki
    
    Args:
        df: zintegrowany DataFrame
    
    Returns:
        DataFrame z dodatkowymi statystykami
    """
    df_stats = df.copy()
    
    # Sortuj według powiatu i roku
    df_stats = df_stats.sort_values(['powiat_code', 'year'])
    
    # Zmiana wskaźnika przestępczości rok do roku
    df_stats['crime_rate_change'] = df_stats.groupby('powiat_code')['crime_rate_per_100k'].diff()
    
    # Procentowa zmiana wskaźnika
    df_stats['crime_rate_change_pct'] = (
        df_stats.groupby('powiat_code')['crime_rate_per_100k'].pct_change() * 100
    ).round(2)
    
    return df_stats


def validate_integration(df: pd.DataFrame) -> dict:
    """
    Waliduje zintegrowany dataset
    
    Args:
        df: zintegrowany DataFrame
    
    Returns:
        Słownik z wynikami walidacji
    """
    results = {
        'total_rows': len(df),
        'unique_powiaty': df['powiat_code'].nunique(),
        'years_range': (df['year'].min(), df['year'].max()),
        'missing_values_crime': df['total_crimes'].isna().sum(),
        'missing_values_population': df['total_population'].isna().sum(),
        'missing_values_crime_rate': df['crime_rate_per_100k'].isna().sum(),
        'avg_crime_rate_per_100k': df['crime_rate_per_100k'].mean(),
        'median_crime_rate_per_100k': df['crime_rate_per_100k'].median(),
        'correlation_density_crime': df['population_density'].corr(df['crime_rate_per_100k'])
    }
    
    return results

print("Tworzenie zintegrowanego datasetu...")

# Ścieżki do plików
crime_path = './output/crime/crime_total_powiaty.csv'
population_density_path = './output/population/population_with_density.csv'
population_gender_path = './output/population/population_gender.csv'
population_age_path = './output/population/population_age_groups.csv'
population_changes_path = './output/population/population_with_changes.csv'

# Utwórz zintegrowany dataset
df = create_integrated_dataset(
    crime_path,
    population_density_path,
    population_gender_path,
    population_age_path,
    population_changes_path
)

print(f"Rozmiar zintegrowanego datasetu: {df.shape}")
print(f"\nKolumny ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# Oblicz dodatkowe statystyki
print("\n" + "="*80)
print("Obliczanie dodatkowych statystyk...")
df_with_stats = calculate_statistics(df)

# Walidacja
print("\n" + "="*80)
print("Walidacja zintegrowanego datasetu:")
validation = validate_integration(df_with_stats)
for key, value in validation.items():
    print(f"  {key}: {value}")

# Przykładowe analizy
print("\n" + "="*80)
print("Statystyki wskaźnika przestępczości (na 100k):")
print(df_with_stats['crime_rate_per_100k'].describe())

print("\n" + "="*80)
print("Wskaźnik przestępczości według kategorii urbanizacji (2024):")
df_2024 = df_with_stats[df_with_stats['year'] == 2024]
print(df_2024.groupby('urbanization_category')['crime_rate_per_100k'].agg(['mean', 'median', 'count']))

print("\n" + "="*80)
print("Top 10 powiatów z najwyższym wskaźnikiem przestępczości (2024):")
top10 = df_2024.nlargest(10, 'crime_rate_per_100k')[['powiat_name', 'crime_rate_per_100k', 
                                                        'population_density', 'urbanization_category']]
print(top10)

print("\n" + "="*80)
print("Top 10 powiatów z najniższym wskaźnikiem przestępczości (2024):")
bottom10 = df_2024.nsmallest(10, 'crime_rate_per_100k')[['powiat_name', 'crime_rate_per_100k', 
                                                            'population_density', 'urbanization_category']]
print(bottom10)

# Zapisz do plików
print("\n" + "="*80)
print("Zapisywanie do plików...")
df_with_stats.to_csv('./output/integrated_crime_data.csv', index=False)

# Dodatkowo: tylko 2024 rok (dla szybkich analiz)
df_2024.to_csv('./output/integrated_crime_data_2024.csv', index=False)

print("\n✓ Gotowe!")
print("\nUzyskane pliki:")
print("  - integrated_crime_data.csv (pełny dataset 2013-2024)")
print("  - integrated_crime_data_2024.csv (tylko 2024 rok)")
print("\nGotowe do:")
print("  - Eksploracyjnej analizy danych (EDA)")
print("  - Budowy modelu ML")
print("  - Tworzenia wizualizacji")