import pandas as pd

# Load the Excel file
excel_file = 'new_data/GDSC/GDSC2_fitted_dose_response_27Oct23.xlsx'  # replace with your Excel file path
sheet_name = 'Sheet 1'           # replace with the sheet name if needed

# Read the Excel sheet into a DataFrame
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Save the DataFrame as a CSV file
csv_file = 'new_data/GDSC/drug_response_data.csv'    # replace with your desired CSV file path
df.to_csv(csv_file, index=False)

print(f"Excel file '{excel_file}' has been converted to CSV '{csv_file}'")