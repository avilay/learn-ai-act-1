# Datasets
To exlore any new charting library I need sample datasets. Ideally a single dataset that has the following:

  * At least 4 numerical (float) fields
  * At least 2 categorical fields (including the target)
  * Timeseries

If I don't find a single such dataset, then I need three datasets with the following characteristics:

  * Has 1 numerical and 2 categorical fields
  * Is a simple timeseries
  * Has 4 numerical fields

## Candidate Datasets

### House Sales in King County
This [Kaggle dataset](https://www.kaggle.com/harlfoxem/housesalesprediction) has 21,600 rows.

#### Columns
  * date: Timeseries (datetime)
  * price: Numerical (int)
  * bedrooms: Numerical/Categorical (int)
  * bathrooms: Numerical (float)
  * sqtf_living: Numerical (int)
  * sqft_lot: Numerical (int)
  * floors: Categorical (int)
  * waterfront: Categorical (int)
  * view: Categorical (int)
