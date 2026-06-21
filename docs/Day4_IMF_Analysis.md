# Day 4 - Interplanetary Magnetic Field (IMF) Analysis

## Objective

The objective of Day 4 was to move beyond Solar Wind Plasma analysis and begin studying the magnetic field carried by the solar wind.

Day 3 answered:

> What material is arriving from the Sun?

Day 4 attempted to answer:

> What magnetic field is arriving with that material?

This is an important transition because Earth's response depends heavily on the magnetic field orientation embedded within the solar wind.

---

# Theoretical Concepts Covered

## Interplanetary Magnetic Field (IMF)

The IMF is the magnetic field carried outward from the Sun by the solar wind.

Conceptually:

Solar Wind = Plasma + Magnetic Field

The solar wind does not only transport particles; it also transports magnetic fields through interplanetary space.

---

## Frozen-In Magnetic Fields

A key concept discussed was the Frozen-In Field approximation.

Because solar plasma is electrically conductive, magnetic field lines move together with the plasma.

Conceptually:

Plasma Moves
↓
Magnetic Field Moves

This explains how magnetic fields generated at the Sun eventually reach Earth.

---

## IMF Components

NOAA provides the magnetic field using GSM coordinates.

The dataset contains:

### Bx

Magnetic field component approximately along the Sun-Earth direction.

### By

Magnetic field component perpendicular to the Sun-Earth line.

### Bz

North-South magnetic field component.

This is the most important variable in operational space weather forecasting.

### Bt

Total magnetic field strength.

Represents the overall magnitude of the IMF.

---

## Why Bz Is Important

One of the major concepts learned today:

Positive Bz:

Earth tends to resist magnetic interaction.

Negative Bz:

Magnetic reconnection becomes easier.

More solar wind energy can enter Earth's magnetosphere.

Thus:

Bz tells us whether Earth should care about the arriving solar wind.

---

# NOAA Dataset Used

Dataset:

mag-7-day.json

Variables:

* time_tag
* bx_gsm
* by_gsm
* bz_gsm
* lon_gsm
* lat_gsm
* bt

Records:

Approximately 9,621 observations.

Time Span:

Approximately 7 days.

---

# Data Engineering Tasks Completed

## API Request

Downloaded NOAA IMF data using:

* requests.get()

---

## JSON Parsing

Converted response to JSON using:

* response.json()

---

## DataFrame Creation

Created Pandas DataFrame using:

* pd.DataFrame()

---

## Data Type Conversion

Converted:

* time_tag → datetime
* bx_gsm → float
* by_gsm → float
* bz_gsm → float
* lon_gsm → float
* lat_gsm → float
* bt → float

---

# Descriptive Statistics

## Bx

Average:

-0.27 nT

Range:

-9.65 to +6.28 nT

---

## By

Average:

0.75 nT

Range:

-8.29 to +9.93 nT

---

## Bz

Average:

0.56 nT

Range:

-7.51 to +11.40 nT

---

## Bt

Average:

6.00 nT

Range:

0.62 to 12.00 nT

---

# Key Event Investigations

## Strongest Southward IMF Event

Question Asked:

What was the most negative Bz value observed?

Code Used:

df.loc[df["bz_gsm"].idxmin()]

Result:

Time:

2026-06-16 13:47

Values:

* Bx = 1.73
* By = -0.60
* Bz = -7.51
* Bt = 7.75

Interpretation:

This was the most geoeffective magnetic orientation observed during the week.

---

## Strongest IMF Event

Question Asked:

What was the strongest overall magnetic field observed?

Code Used:

df.loc[df["bt"].idxmax()]

Result:

Time:

2026-06-17 05:32

Values:

* Bx = 3.10
* By = 8.77
* Bz = 7.58
* Bt = 12.00

Interpretation:

The strongest magnetic field event was northward rather than southward.

Important lesson:

Strong magnetic field does not automatically imply high geomagnetic storm potential.

Bz orientation remains critical.

---

# Positive vs Negative Bz Analysis

Question Asked:

How often was Bz positive versus negative?

Code Used:

(df["bz_gsm"] < 0).sum()

(df["bz_gsm"] > 0).sum()

Results:

Negative Bz:

4,194 measurements

Positive Bz:

5,408 measurements

Interpretation:

Approximately 44% of measurements were southward.

Approximately 56% were northward.

The IMF spent a substantial portion of the week in potentially geoeffective conditions.

---

# Visualization Analysis

## Bz Time Series

Created Bz versus Time graph.

Observations:

* Frequent crossings of zero.
* Multiple southward IMF intervals.
* No extremely large negative excursions.
* Most negative event near -7.5 nT.

---

## Bt Time Series

Created Bt versus Time graph.

Observations:

* Typical IMF conditions.
* Peak field strength approximately 12 nT.
* No extreme magnetic storms evident from Bt alone.

---

# Correlation Analysis

Correlation Matrix Examined:

Variables:

* Bx
* By
* Bz
* Bt

Major Finding:

Bz showed very weak correlation with both Bx and By.

Interpretation:

Bz behaves largely independently and must be monitored separately.

---

# Solar Wind + IMF Integration

Question Asked:

Should Solar Wind and IMF datasets be merged?

Answer:

Yes.

Both datasets share a common timestamp and can be merged using:

time_tag

---

## Merged Dataset Variables

Solar Wind:

* density
* speed
* temperature

IMF:

* bx_gsm
* by_gsm
* bz_gsm
* bt

---

# Cross-Dataset Correlation Analysis

Question Asked:

Do Solar Wind properties affect Bz?

Results:

Speed vs Bz:

0.046

Density vs Bz:

0.251

Temperature vs Bz:

-0.181

Interpretation:

Solar Wind plasma properties showed little relationship with Bz orientation.

Bz behaves relatively independently.

---

## Strongest Relationship Observed

Speed vs Temperature:

0.81

Interpretation:

Faster solar wind generally contains hotter plasma.

This was the strongest relationship discovered across all analyses.

---

# Forecasting Questions Explored

Several conceptual forecasting scenarios were discussed.

Examples:

1. High Speed + High Temperature + Positive Bz
2. High Speed + High Temperature + Negative Bz
3. Low Speed + Low Temperature + Positive Bz
4. Low Speed + Low Temperature + Negative Bz

Major Conclusion:

Negative Bz acts like an open door for energy transfer.

However, the amount of solar wind energy available still matters.

Both plasma conditions and magnetic orientation must be considered together.

---

# Advanced Question: Duration

Question Asked:

The dataset contains Bz values, but where is duration?

Answer:

Duration is not directly provided.

It must be derived from timestamps.

Example:

Consecutive negative Bz values can be grouped into intervals.

This allows measurement of:

* Longest negative Bz interval
* Longest Bz < -3 interval
* Longest Bz < -5 interval
* Total southward IMF duration

This became a Day 4 Bonus Analysis topic.

---

# Major Lessons Learned

1. Solar wind carries magnetic fields.
2. Bz is the most important IMF component.
3. Negative Bz increases the likelihood of energy transfer into Earth's magnetosphere.
4. Strong magnetic fields do not automatically imply storms.
5. Bz direction matters as much as magnetic field strength.
6. Solar wind speed strongly correlates with temperature.
7. Solar wind properties show little relationship with Bz orientation.
8. Duration of southward IMF is an important forecasting variable.
9. IMF analysis is the bridge between solar physics and geomagnetic storm forecasting.

---

# Day 4 Deliverables

Completed:

* NOAA IMF Dataset Download
* JSON Inspection
* DataFrame Creation
* Data Type Conversion
* Bx Analysis
* By Analysis
* Bz Analysis
* Bt Analysis
* Event Investigation
* Visualization
* Correlation Analysis
* Solar Wind + IMF Merge
* Cross-Dataset Analysis
* Forecasting Discussion
* Bz Persistence Planning

---

# Next Step

Dataset 3:

Kp Index

Primary Question:

Given Solar Wind and IMF conditions,

How did Earth actually respond?
