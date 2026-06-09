import pandas as pd
import os

# Paths to the Excel files
f1 = 'data/excel/AlbufeirasMaranhao_18-07-2025.xlsx'
f2 = 'data/excel/Historico_2005_2025_V15NOV2025.xlsx'

print("Loading AlbufeirasMaranhao_18-07-2025.xlsx...")
df1 = pd.read_excel(f1, sheet_name=0)
# Extract Data (col 0) and Cota (col 5)
df1_extracted = pd.DataFrame({
    'date': pd.to_datetime(df1.iloc[:, 0], errors='coerce'),
    'cota': pd.to_numeric(df1.iloc[:, 5], errors='coerce')
})

print("Loading Historico_2005_2025_V15NOV2025.xlsx...")
df2 = pd.read_excel(f2, sheet_name=0)
# Filter for Maranhão where column 0 (Barragem) matches 'Maranhão' (or contains 'Maranh')
df2_mar = df2[df2.iloc[:, 0].astype(str).str.contains('Maranh', na=False)]
# Extract Data (col 1) and Cota (col 2)
df2_extracted = pd.DataFrame({
    'date': pd.to_datetime(df2_mar.iloc[:, 1], errors='coerce'),
    'cota': pd.to_numeric(df2_mar.iloc[:, 2], errors='coerce')
})

# Concatenate both datasets
combined = pd.concat([df1_extracted, df2_extracted], ignore_index=True)

# Drop rows where date or cota is null/invalid
combined = combined.dropna(subset=['date', 'cota'])

# Format dates as YYYY-MM-DD
combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')

# Drop duplicate dates, keeping the first occurrence (since we verified they are identical)
combined = combined.drop_duplicates(subset=['date'])

# Sort chronologically by date
combined = combined.sort_values(by='date').reset_index(drop=True)

# Save to CSV
output_path = 'cota_maranhao.csv'
combined.to_csv(output_path, index=False)
print(f"\nSuccessfully generated CSV at: {os.path.abspath(output_path)}")
print(f"Total entries: {len(combined)}")
print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
print("\nFirst 10 rows:")
print(combined.head(10))
print("\nLast 10 rows:")
print(combined.tail(10))
