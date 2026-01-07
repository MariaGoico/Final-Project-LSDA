Group information.
We are Group 3, formed by Joaquin Orradre, Paula Pina, and María Goicoechea.

Dataset.
The dataset chosen for this project is the Global Historical Climatology Network – Daily (GHCN-Daily), provided by NOAA. This dataset contains daily climate observations collected from thousands of land-based weather stations worldwide. The documentation is publicly available through NOAA, and the data itself can be accessed through their official archive. Each daily record includes meteorological variables such as temperature, precipitation, wind speed, pressure, visibility, geographic location (latitude, longitude, elevation), and snow depth on the ground (SNDP). The dataset is well suited for this project due to its large scale, long temporal coverage, and rich set of explanatory variables, which allow both predictive modeling and scalability analysis.

Problem / objective.
The main objective of our project is to predict snow depth on the ground (SNDP) using daily meteorological and geographic variables (a regression problem). We aim to understand which factors most strongly influence snow accumulation and how well snow depth can be predicted based on readily available climate measurements.

Approach / methodology.
We plan to perform exploratory data analysis to identify relevant patterns and relationships between snow depth and other variables. Preliminary correlation analysis (considering only days with SNDP > 0) suggests moderate correlation with temperature (0.4540), weak correlation with precipitation (0.0429), latitude (−0.2122), and elevation (−0.0048). Additional features, such as the month of the year and other weather indicators, will be explored. 

Scalability.
Scalability is a central aspect of this project. The GHCN-Daily dataset consists of very large files, with the number of active stations in later years ranging from approximately 7,000 to 9,000 stations, each providing daily measurements. This results in millions of records per year. To study size-up, we will progressively increase the data volume by incorporating additional years, a larger number of stations, and finer temporal resolution. In practice, this will be done by subsampling the dataset in a controlled manner, for example by using subsets of stations, restricting the analysis to selected year ranges, or reducing temporal resolution by taking one out of every two (or more) days. Of course, from the beginning we will not take into account every year, we have thought to take the last 15 years since it looks like there are more than enough data. (Look at attached image)

Expected outcomes (optional).
We expect to identify the most relevant predictors of snow depth, those being temperature variables, the precipitation in relation to the temperature (low temperature + precipitation, seasonality and geographical features) evaluate the performance of our predictive models.