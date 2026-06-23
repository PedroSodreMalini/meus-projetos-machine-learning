import sweetviz as sv
import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

sv_churn_report = sv.analyze(df_churn, target_feat="Churn")
sv_churn_report.show_html(filepath="./automatizando-eda/report-churn.html", open_browser=True)