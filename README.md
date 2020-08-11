Technology used: Python, Jupyter Notebooks, Machine Learning Algorithms

An exciting challenge, we are given data from the Kepler mission. Kepler was NASA's groundbreaking exoplanetary survey project that finally ended in 2018.
Kepler looked in the direction of the constellation Cyngus for evidence of exoplanets, that is, planets that orbit other stars. This is a highly difficult
challenge: without going to much into the maths, consider how faint distant stars are, and consider how much more massive they are than any other body in
their orbit. And yet, in the many years Kepler operated, Kepler found 2600 exoplanets, some of which have signs of being terrestrial planets. Kepler has 
paved the way for other projects, such as TESS.

Our task was to create a few models and see how well they do. It was a very open-ended project, but highly enjoyable. I decided to tackle this problem
with three models, and changing parameters to find the best results. The three models are:

-A logistical model (Model 1) that uses gridsearchCV to tune the parameters,
-A SCV model (Model 2) that also uses gridsearchCV to tune the parameters, and
-A deep-learning model (Model 3) that uses one-hot encoding and keras to tune and test.

For the first test, I chose to use the following columns of data for my features:
-koi_fpflag_nt: Not Transit-Like Flag. *The light curve is not consistent with that of a transiting planet.

-koi_fpflag_ss: Stellar Eclipse Flag. *A KOI that is observed to have a significant secondary event, transit shape, 
or out-of-eclipse variability, which indicates that the transit-like event is most likely caused by an eclipsing binary.

-koi_fpflag_co: Centroid Offset Flag. *The source of the signal is from a nearby star, as inferred by measuring the 
centroid location of the image both in and out of transit.

-koi_fpflag_ec: Ephemeris Match Indicates Contamination Flag. *The KOI shares the same period and epoch as another object 
and is judged to be the result of flux contamination in the aperture or electronic crosstalk.

-koi_prad: Planetary Radius (Earth radii). *The radius of the planet. Planetary radius is the product of the planet star 
radius ratio and the stellar radius.

The first four were chosen since they are a flag to check and test to see if the KOI matches criteria necessary to fit the profile of an exoplanet. The
planetary radii was chosen based on whether the planet might be terrestrial or not (the larger a planetary radii, the least likely it is to be
terrestrial).

The results are as follows (rounded to three decimal places):
Model 1: 0.760
Model 2: 0.791
Model 3: Loss: 0.367, Accuracy: 0.764

We see that with the data selected, the Support vector machine model performed the best at 79.1% accuracy, while the LinearRegression and Deep Learning models
did worse, at 76% accuracy.

I decided to try running the models again, this time only focusing only on the KOI flag data and nothing else. Each model was run again, with the following results:

Model 1: 0.747
Model 2: 0.747
Model 3: Loss: 0.367, Accuracy: 0.764

Dropping a column of data from the fit leads to a worse accuracy in both the Logistical and SVC model, while the deep learning model performed the same. Dropping data didn't result in anything new, so I decided to add back in a fifth column: this time, koi_teq, the approximate tempearture of the planet in Kelvins. I was interested in seeing how my models would stand up to extreme condition planets: for example, hot Jupiters, gas giants that orbited far closer to the sun than predicted before their discovery a few years ago. Adding this data in, and running the models for the last time, the results were:

Model 1: 0.755
Model 2: 0.649
Model 3: Loss: 0.364, Accuracy: 0.773

Model 1 does better than the second attempt, though still slightly worse than the first run. Model 2 has a relatively precipitous drop, hovering around 65% accuracy. The third model does better than the previous two attempts, at 77% accuracy.

Would any of these models be a good tool at predicting exoplanets based on Kepler data? With the exception of the second run of the SVC model, they all had accuracy values slightly below or above 75%. These would be decent "first-attempt" models, but I would not try and publish the results (especially without more scientific rigor). What could I do to make the results better, given more time? Now that I have tried different fits, the next step would be to use the same models, but start tuning the parameters instead. What happens when I add (or remove) more nodes in the deep learning model? What if I change the parameters of the gridsearchCV in the Logistical model? What if I use a different kernel in the SVC model, and/or change the gridsearchCV parameters as well?

I plan to explore these avenues on my own time. I would feel I reached a satisfactory completion if I were to obtain scores on any (or all) of the models above 96%. Part of the fun of learning ML is figuring out the best methods by trial-and-error. I will do a bit of study/research, but I will also just play around and see what works best. 
