# Day 3 - Solar Wind Dataset Analysis

## Dataset

NOAA Solar Wind Plasma (7-Day Dataset)

Variables:

* time_tag
* density
* speed
* temperature

Rows Analyzed:

~9200 observations

---

# What Solar Wind Is

Solar wind is a continuous flow of plasma escaping from the Sun.

Contains:

* Protons
* Electrons
* Helium Ions

Solar wind carries:

* Matter
* Energy
* Magnetic Fields (IMF)

toward Earth.

---

# Dataset Findings

## Density

Measures:

Number of solar wind particles per unit volume.

Observed Range:

Min: 0.09

Max: 17.49

Average: ~6.53

Observation:

Solar wind density changes significantly over time.

Some periods are very sparse while others contain dense plasma streams.

---

## Speed

Measures:

Velocity of solar wind.

Observed Range:

Min: 356.7 km/s

Max: 606.9 km/s

Average: ~435 km/s

Observation:

Most solar wind remained in the normal speed range.

Occasional high-speed streams were observed.

---

## Temperature

Measures:

Energy of solar wind particles.

Observed Range:

Min: 2,000 K

Max: 552,298 K

Average: ~112,257 K

Observation:

Temperature varies much more dramatically than density and speed.

---

# Relationships Between Variables

## Density vs Speed

Correlation:

-0.189

Observation:

Very weak negative relationship.

Higher speed does not necessarily mean higher density.

Fast solar wind tends to be slightly less dense.

---

## Density vs Temperature

Correlation:

0.345

Observation:

Weak positive relationship.

Higher density often corresponds to somewhat higher temperatures.

---

## Speed vs Temperature

Correlation:

0.522

Observation:

Moderate positive relationship.

Faster solar wind generally tends to be hotter.

This was the strongest relationship found in the dataset.

---

# Interesting Events

## Fastest Solar Wind Event

Time:

2026-06-13 15:59

Values:

Density: 0.27

Speed: 606.9 km/s

Temperature: 8,092 K

Observation:

Highest speed occurred with extremely low density.

---

## Densest Solar Wind Event

Time:

2026-06-17 01:55

Values:

Density: 17.49

Speed: 403.2 km/s

Temperature: 73,222 K

Observation:

Highest density did not correspond to highest speed.

---

## Hottest Solar Wind Event

Time:

2026-06-14 03:33

Values:

Density: 12.85

Speed: 530.7 km/s

Temperature: 552,298 K

Observation:

Highest temperature occurred during relatively fast and dense solar wind conditions.

---

# What I Learned

1. Solar wind is not constant.
2. Density changes significantly over time.
3. Speed changes throughout the week.
4. Temperature shows large variability.
5. Faster solar wind generally tends to be hotter.
6. High density does not necessarily mean high speed.
7. Solar wind consists of plasma carrying matter from the Sun.
8. Solar wind data alone cannot determine geomagnetic storm risk.
9. Magnetic field data (IMF/Bz) is required next.

---

# Next Dataset

IMF (Interplanetary Magnetic Field)

Variables:

* Bx
* By
* Bz

Goal:

Understand how the magnetic field embedded in solar wind affects Earth.
