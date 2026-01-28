import pandas as pd
from agents.indicator_agent import indicator_agent

df = pd.read_csv("data/RELIANCE.csv")
result = indicator_agent(df)

print(result)
