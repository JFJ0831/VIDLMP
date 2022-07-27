# Generative Adversarial Networks

## Was sind Generative Adversarial Networks?
Generative Adversarial Networks, kurz GAN, gehören zu den generativen Modellen. Diese Modelle sind in der Lage neue Daten, also beispielsweise Bilder von fiktiven Gesichtern, zu erstellen.
GANs bestehen aus zwei neuronalen Netzen welche gegeneinander antreten.
Eins der beiden Netze, Generator genannt, versucht dabei echte Daten so gut zu imitieren, dass das andere Teilnetz, der Discriminator, nicht in der Lage ist, die neu generierten Daten von echten Daten zu unterscheiden.

## Wozu werden GANs verwendet
Im ursprünglichen Forschungspaper von Ian Goodfellow et al. wurde ein GAN erstellt, welches Bilder von Zahlen generieren kann, die so aussehen, wie die aus dem MNIST-Datensatz.
Mittlerweile wurden GANs in einer Vielzahl von Anwendungsgebieten erfolgreich eingesetzt. Dazu sei jedoch gesagt, dass die verwendeten Modelle deutlich komplexer geworden sind, als das ursprüngliche.
Die prominenteste Verwendung für GANs ist wohl im Bereich der Bildgenerierung und insbesondere der Generierung von Gesichtern nicht existierender Personen angesiedelt.
In diesem Bereich nehmen die sogenannten StyleGANs eine Vorreiterrolle ein.

## Wie funktionieren GANs?
Im diesem Blog soll ein GAN vorgestellt werden, welches in der Lage ist echt aussehende Bilder von Früchten zu generieren.
Bevor es um den eigentlichen Code geht, muss der theoretische Aufbau und Ablauf erläutert werden.
Wie bei vielen anderen Modellen die sich mit maschinellem Lernen befassen sind oft hunterte oder gar hunterttausende Trainingsdurchläufe notwendig, bis das Modell ein zufriedenstellendes Ergebnis liefert. 
Im Fall von GANs handelt es sich dabei um ein sogenanntes Nash-Gleichgewicht, also einen Zustand, bei dem es sich weder für den Diskriminator, noch für den Generator lohnt sich zu verändern.
Welche Herausforderungen damit verbunden sind, wird später [hier](## Herausforderungen beim GAN-Training und Anpassungsmöglichkeiten) erklärt. Zuerst geht es jetzt um die zwei Phasen eines Trainingsdurchlaufs.

### Erste Phase
Der erste Teil des GANs, der Generator, wird mit einem zufälligen Input gefüttert. In vielen Fällen handelt es sich dabei um gaußsches Rauschen, aber auch andere Verteilungen sind möglich.
Den Input bearbeitet der Generator nun bestenfalls so, dass echt aussehende Bilder von Früchten entstehen. In den ersten Durchläufen wird dies nicht gelingen, der Output wird mehr oder weniger zufällig sein.
Dies liegt dran, dass die Gewichte des neuronalen Netzwerkes zufällig initialisiert wurden und auch noch nicht ausreichend in eine Richtung angepasst werden konnten, die es ermöglicht echt aussehende Daten zu generieren.
Bevor die Gewichte des Generators überhaupt angepasst werden, wird zunächst der Discriminator trainiert.
Dieser wird dazu sowohl mit gerade neu generierten Daten, aber auch mit echten Daten gefüttert. Die Daten kommen dabei jeweils in gleicher Anzahl vor.
Der Discriminator versucht nun die Daten richtig zuzuordnen. Wie erfolgreich diese geschieht, lässt sich mithilfe einer Verlustfunktion errechnen.
Nun gilt es festzustellen, welche Kantengewichte wie angepasst werden müssen, um den Verlust zu reduzieren.
Dazu werden unter Zuhilfenahme von Backpropagation entsprechende Gradienten berechnet und dann die Gewichte der Kanten in die entgegengesetzte Richtung der Gradienten angepasst (-> stochastischer Gradientenabstieg). 
Zu Beachten ist, dass während des Trainings des Discriminators die Gewichte des Generators nicht verändert werden. Dieser Teil des GANs wird in diesem Schritt sogar komplett ausgeklammert.

### Zweite Phase
Sind die Gewichte des Discriminators aktualisiert worden, so widmet sich der nächste Schritt dem Training des Generators.
Auch hier wird der Generator zuerst mit Zufallseingaben beschickt, welche dann zu möglichst realistischen Daten modifiziert werden.
Die Unterscheidung zum ersten Schritt liegt unter anderem darin, dass der Discriminator jetzt nicht sowohl mit echten als auch unechten Daten gefüttert wird, sondern nur mit unechten.
Der Discriminator versucht nun möglichst alle Daten als unecht zu erkennen. Auch hier wird ein Verlust berechnet, wobei die Funktion eine andere sein kann als beim Training des Discriminators. 
Der Generator wird bestraft, wenn der Discriminator ein generiertes Bild als solches erkennt.
Um die Gewichte des Generators anzupassen, muss die Backpropagation des Verlustes folglich über das gesamte GAN erfolgen.
Negative Auswirkungen auf die Leistung des Discriminators lassen sich unterbinden, indem dessen Gewichte zuvor festgesetzt werden und für diesen Schritt unveränderlich sind.
Nach Aktualisierung der Generatorgewichte kann ein neuer Durchlauf begonnen werden.

## Herausforderungen beim GAN-Training und Anpassungsmöglichkeiten
Wenn gewünscht, ist es mithilfe von Hyperparametern möglich den ersten und/oder den zweiten Schritt mehrfach hintereinander auszuführen.
Wenn pro Trainingsdurchlauf beispielsweise der erste Schritt häufiger durchlaufen wird als der zweite, so erhält man einen Diskriminator, der deutlich besser echte Daten von unechten Daten des aktuellen Generators unterscheiden kann.
Ob dies negative oder positive Auswirkungen bei der Erreichung eines guten Generators hat, muss für jedes GAN ausprobiert werden.



## Mathematischer Ablauf
Der Generator $G$ soll eine Wahrscheinlichkeitsdichtefunktion $p_g$ erstellen, sodass $G(z)$ mit der gleichen Wahrscheinlichkeit in einem Intervall $[a, b]$ liegt, wie die Zufallsvariable $x$ in einem Intervall $[a, b]$ der Wahrscheinlichkeitsdichtefunktion $p_{data}$.
Der Diskriminator $D$ entscheidet, ob $p_g=p_{data}$, oder nicht. In ersterem Fall ist es dem Generator $G$ gelungen den Diskriminator $D$ zu täuschen.
Der Generator $G$ erhält, wie als Eingabe Werte $z$ aus einer beliebigen Verteilung. Oft handelt es sich dabei, wie bereits geschrieben, um Gaußsches Rauschen, also die Normalverteilung.


Die echten Daten werden mit dem Label $y=1$ und die unechten Daten mit dem Label $y=0$ übergeben.

Auch wird als Label trotz unechter Daten $y=1$ übermittelt.


<p align="center">
	<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_1.png" title="Abbildung 1" width="256"/>
	<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_2.png" title="Abbildung 2" width="256"/>
	<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_3.png" title="Abbildung 3" width="256"/>
	<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_4.png" title="Abbildung 4" width="256"/>
</p>
