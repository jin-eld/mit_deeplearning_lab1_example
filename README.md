# Implementation of the MIT Deep Learning Lab 1 Music Generation excercise in Rust / Burn


The implementation is done purely for learning purposes and is based on:
https://github.com/aamini/introtodeeplearning/blob/2023/lab1/solutions/Part2_Music_Generation_Solution.ipynb

The lecture associated with the above example can be viewed on YouTube:
https://www.youtube.com/watch?v=dqoEU9Ac3ek

The dataset originates from:
https://github.com/aamini/introtodeeplearning/blob/2023/mitdeeplearning/data/irish.abc, Licensed under the MIT License, © MIT Introduction to Deep Learning,
http://introtodeeplearning.com


This is a work in progress and does not function correctly yet. Inference
produces garbage instead of nicely structured ABC notation.

At this point there is only rudimentary error checking and the application will
panic if something is not as expected, this will be improved later once the
actual ML part is fixed.
