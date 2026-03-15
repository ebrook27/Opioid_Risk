import csv
import sys

path = 'data/Raw/CAEMP25N__ALL_AREAS_2001_2022.csv'

selected_rows = 0
total_D = 0

try:
    with open(path, newline='', encoding='latin-1') as f:
        reader = csv.reader(f)
        headers = next(reader)

        # detect indices
        if 'IndustryClassification' not in headers:
            print('Error: IndustryClassification column not found. Available headers:')
            print(headers)
            sys.exit(1)
        idx_ind = headers.index('IndustryClassification')

        # GeoName is the place name (state/county)
        if 'GeoName' not in headers:
            print('Error: GeoName column not found. Available headers:')
            print(headers)
            sys.exit(1)
        idx_geo = headers.index('GeoName')

        # Year columns are those that look like '2001','2002',... detect them
        year_cols = [(i, h) for i, h in enumerate(headers) if h.strip().isdigit()]
        years = [h for (_, h) in year_cols]

        year_counts = {y: 0 for y in years}
        state_counts = {}

        for row in reader:
            if len(row) <= idx_ind:
                continue
            if row[idx_ind].strip() == '62':
                selected_rows += 1

                # count '(D)' across year columns
                for i, y in year_cols:
                    if i < len(row) and row[i].strip() == '(D)':
                        year_counts[y] += 1
                        total_D += 1

                # extract state from GeoName (heuristic)
                geo = row[idx_geo].strip()
                state_key = None
                if ',' in geo:
                    # common format: 'County Name, ST' or 'Place, ST'
                    parts = [p.strip() for p in geo.split(',')]
                    state_key = parts[-1]
                else:
                    # could be 'United States' or state name
                    state_key = geo

                if state_key == '':
                    state_key = 'UNKNOWN'

                state_counts[state_key] = state_counts.get(state_key, 0) + sum(1 for i, _ in year_cols if i < len(row) and row[i].strip() == '(D)')

except FileNotFoundError:
    print(f"Error: file not found at {path}")
    sys.exit(1)

print(f"Selected rows: {selected_rows}")
print(f"Total number of '(D)' entries in selected rows: {total_D}")

print('\n(D) counts by year:')
for y in sorted(year_counts.keys()):
    print(f"{y}: {year_counts[y]}")

print('\nTop states by (D) count (descending):')
for state, cnt in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:50]:
    print(f"{state}: {cnt}")
