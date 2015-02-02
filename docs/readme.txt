Assumptions:
- All fields have been treated equally - same crops, same inputs
- All fields have been treated uniformly - farmers not already trying to compensate for these factors
- No penalty for switching prescriptions frequently
- Aspect only meaningful in presence of slope
- Missing data missing for a reason - unworkable land, etc.  No attempt to interpolate.
- Lots of the differences in LAI are clearly (by satellite imagery) just due to what parts of the field were planted
    - see 8, 16, 17, 57, 68, 97, 150, 161, 175, 179
- Productivity zones = low, med, high.  (kinda silly if range in predicted yield is low)
- WDRVI/LAI not appropriate model input
    - measurements for current year will lag nutrient treatment
    - historical data a) not available or b) not useful predictor because of crop rotation, etc.
- Goal is not to estimate LAI itself, rather management zones

Interesting cases
25, 64, 68, 71, 91, 97, 102, 105, 114, 117, 118, 135, 160, 161, 171, 172, 186

Observations:
- Often soil type is correlated with slope/aspect/curvature - 105, 149, 172
- Landsat pix size: ~30 meter pixel (1/4 acre) -- http://coast.noaa.gov/digitalcoast/data/landsat

Notes:
- Soil type mapped to soil texture triangle.  A lot more information is available about these soil types, but not easy to scrape
- Any soil types that couldn't be classified according to soil texture triangle, e.g., 'Dickman' removed


Thought experiment:
- Two independently distributed RVs
- Transformed into discrete RVs: low, middle, high
- Given thresholds of 33 and 66 pctile:

        .333    .333    .333
.333    .111    .111    .111
.333    .111    .111    .111
.333    .111    .111    .111
    expected concordance: .333 (sum diagonal) / sum(all)


- Given thresholds of 25 and 75 pctile:
        .25     .5      .25
.25     .0625   .125    .0625
.5      .125    .25     .125
.25     .0625   .125    .0625
    expected concordance: .375

- Given thresholds of 10 and 90pctile:
        .10     .9      .10
.10     .001    .09     .001
.9      .09     .81     .09
.1      .001    .09     .001
    expected concordance: .812

The wider we make the middle band, the higher the expected concordance