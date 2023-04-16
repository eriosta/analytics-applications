import pandas_profiling
import pandas as pd

df = pd.read_csv('customers/acquisitionRetention.csv')

# Assuming final_merged_df is your dataframe
profile = pandas_profiling.ProfileReport(df,
                                         title="Descriptive Statistics Report",
                                         explorative=True,
                                         html={'style': {'full_width': True}},
                                         dark_mode=True
                                         )

# Save the report as an HTML file
profile.to_file("descriptive_statistics_report.html")