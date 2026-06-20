import requests
import pandas as pd

url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"

response = requests.get(url)


#print("Status Code:", response.status_code)

#if response.status_code == 200:
 #   data = response.json()
  #  print(data)
#else:
 #   print("Error fetching data from NOAA API")

data = response.json()

#print(type(data))

#print(data[0])  # Print the first entry in the data list

#print("Rows:", len(data))
#print("Header:", data[0])
#print("First Record:", data[1])

headers = data[0]
rows = data[1:]

df = pd.DataFrame(rows, columns=headers)

df["density"] = pd.to_numeric(df["density"])
df["speed"] = pd.to_numeric(df["speed"])
df["temperature"] = pd.to_numeric(df["temperature"])

df["time_tag"] = pd.to_datetime(df["time_tag"])

print(df.head())  # Display the first few rows of the DataFrame

# print(df.info())  # Display information about the DataFrame

# print(df.describe())  # Display summary statistics of the DataFrame

#print(df["density"].describe())  # Display summary statistics for the 'density' column

#import plotly.express as px

#fig = px.line(
#    df,
#    x="time_tag",
#    y = "density",
#    title = "Solar Wind Density - Last 7 Days"
#)

#fig.show()

# print(df[["density", "speed"]].corr())

print(df.loc[df["speed"].idxmax()])
print(df.loc[df["density"].idxmax()])
print(df.loc[df["temperature"].idxmax()])

print(
    df[["density", "speed", "temperature"]].corr()
)