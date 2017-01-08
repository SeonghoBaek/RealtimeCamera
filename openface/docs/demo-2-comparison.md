## Demo 2: Comparing two images
Released by [Brandon Amos](http://bamos.github.io) on 2015-10-13.

---

The [comparison demo](https://github.com/cmusatyalab/openface/blob/master/demos/compare.py) outputs the predicted similarity
score of two faces by computing the squared L2 distance between
their representations.
A lower score indicates two faces are more likely of the same person.
Since the representations are on the unit hypersphere, the
scores range from 0 (the same picture) to 4.0.
The following distances between images of John Lennon and
Eric Clapton were generated with
`./demos/compare.py images/examples/{lennon*,clapton*}`.

| Lennon 1 | Lennon 2 | Clapton 1 | Clapton 2 |
|---|---|---|---|
| <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-1.jpg' width='200px'></img> | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-2.jpg' width='200px'></img> | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/clapton-1.jpg' width='200px'></img> | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/clapton-2.jpg' width='200px'></img> |

The following table shows that a distance threshold of `0.99` would
distinguish these two people.
In practice, further experimentation should be done on the distance threshold.
On our LFW experiments, the mean threshold across multiple
experiments is `0.99`,
see [accuracies.txt](https://github.com/cmusatyalab/openface/blob/master/evaluation/lfw.nn4.small2.v1/accuracies.txt).

| Image 1 | Image 2 | Distance |
|---|---|---|
| Lennon 1 | Lennon 2 | 0.763 |
| Lennon 1 | Clapton 1 | 1.132 |
| Lennon 1 | Clapton 2 | 1.145 |
| Lennon 2 | Clapton 1 | 1.447 |
| Lennon 2 | Clapton 2 | 1.521 |
| Clapton 1 | Clapton 2 | 0.318 |
