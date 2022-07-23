# Generative Adversarial Networks

## Was sind Generative Adversarial Networks?
Generative Adversarial Networks, kurz GAN, gehören zu den generativen Modellen. Diese Modelle sind in der Lage neue Daten, also beispielsweise Bilder von fiktiven Gesichtern, zu erstellen.
GANs bestehen aus zwei neuronalen Netzwerken welche gegeneinander antreten.
Eins der beiden Netzwerke, Generator genannt, versucht dabei echte Daten so gut zu imitieren, dass der andere Netzwerkteil, der Discriminator, nicht in der Lage ist, die neu generierten Daten von echten Daten zu unterscheiden.

## Wozu werden GANs verwendet?
Im ursprünglichen Forschungspaper von Ian Goodwill wurde ein GAN erstellt, welches Bilder von Zahlen des MNIST-Datensatzes generieren konnte.
Mittlerweile wurden GANs in einer Vielzahl von Anwendungsgebieten erfolgreich eingesetzt. Dazu sei jedoch gesagt, dass die verwendeten Modelle deutlich komplexer geworden sind, als das ursprüngliche.
Die am weitesten verbreitete Verwendung für GANs bleibt jedoch im Bereich der Bildgenerierung und insbesondere der Generierung von Gesichtern nicht existierender Personen.
In die sem Bereich nehmen die sogenannten StyleGANs eine Vorreiterrolle ein.

## Wie funktionieren GANs?
Im folgenden soll ein GAN gebaut werden, welches in der Lage ist echt aussehende Bilder von Früchten zu generieren.
Bevor der eigentliche Code geschrieben wird, muss der theoretische Aufbau und Ablauf erläutert werden.
Der erste Teil des GANs, der Generator wird mit einem zufälligen Input gefüttert. Meist handelt es sich dabei um gaussches Rauschen, aber auch andere Verteilungen sind möglich.
Nun versucht der Generator hieraus Bilder von Früchten zu generieren. In den ersten Trainingsdurchläufen wird dies nicht gelingen, der Output wird mehr oder weniger zufällig sein.
Dies liegt dran, dass die Gewichte des neuronalen Netzwerkes zufällig initialisiert wurden und auch noch nicht ausreichend in eine Richtung angepasst werden konnten, die es ermöglicht echt aussehende Daten zu generieren.
Bevor die Gewichte des Generators überhaupt angepasst werden, wird zunächst der Discriminator trainiert.
Dieser wird dazu sowohl mit gerade neu generierten Daten, aber auch mit echten Daten gefüttert. Die Daten kommen dabei jeweils in gleicher Anzahl vor.
Die echten Daten werden mit dem Label $y=1$ und die unechten Daten mit dem Label $y=0$ übergeben.
