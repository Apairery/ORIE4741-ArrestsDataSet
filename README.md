### ORIE4741-ArrestsDataSet

# Examining the NYPD Arrests Dataset

Group members: Mengjia Xia (mx233), Binxin Liu (bl642)

We will use [Stop, Question and Frisk Data](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page), which records every stop, question and frisk effected in NYC ranging from 01/01/2003 to 12/31/2018. Each row in the data records details of every stop, question and frisk and whether the suspect(stopped, questioned and frisked) was eventually arrested. There are rich potential features in this data set, such as dates, times, locations, physical features of the suspect, crime suspected and so on. Features in data set can vary from year to year, but they share the most basic and potentially important features in common.

Since the there is clear label in this data set, whether the suspect was eventually arrested, it is natural for us to do a classification with the dataset. More specifically, we would lile to predict whether the suspect will be arrested or not based on the features used.

This prediction is meaningful as it explore whether some discrimation exists when making the arrestment decision. Moreover, this prediction model can be a guide indicating whether there are avoidable mistakes in the desicion.

Since the large size of this data set(roughly 2 GB), we will not develop a model using all the data. Instead, an exploratary data analysis and an elementary prediction model will be based on the data only in 2016. Then, we will try to generalize our model to a non-stantionary one.

See our [Project Proposal](https://github.com/s2c/ORIE4741-ArrestsDataSet/blob/master/project_proposal.pdf) for more detailed information.

This project is for [Fall 2019 ORIE 4741](https://github.com/ORIE4741/ProjectsFall2019).

### Dataset
https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u

