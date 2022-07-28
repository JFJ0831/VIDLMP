# Generative Adversarial Networks

## Was sind Generative Adversarial Networks?
Generative Adversarial Networks, kurz GAN, gehören zu den generativen Modellen. Diese Modelle sind in der Lage neue Daten, also beispielsweise Bilder von fiktiven Gesichtern, zu erstellen.
GANs bestehen aus zwei neuronalen Netzen welche gegeneinander antreten.
Eins der beiden Netze, Generator genannt, versucht dabei echte Daten so gut zu imitieren, dass das andere Teilnetz, der Discriminator, nicht in der Lage ist, die neu generierten Daten von echten Daten zu unterscheiden.

## Wozu werden GANs verwendet?
Im ursprünglichen Forschungspaper von Ian Goodfellow et al. wurde ein GAN erstellt, welches Bilder von Zahlen generieren kann, die so aussehen, wie die aus dem MNIST-Datensatz.
Mittlerweile wurden GANs in einer Vielzahl von Anwendungsgebieten erfolgreich eingesetzt. Dazu sei jedoch gesagt, dass die verwendeten Modelle deutlich komplexer geworden sind, als das ursprüngliche.
Die prominenteste Verwendung für GANs ist wohl im Bereich der Bildgenerierung und insbesondere der Generierung von Gesichtern nicht existierender Personen angesiedelt.
In diesem Bereich nehmen die sogenannten StyleGANs eine Vorreiterrolle ein.

## Wie funktionieren GANs?
Im diesem Blog soll ein GAN vorgestellt werden, welches in der Lage ist echt aussehende Bilder von Früchten zu generieren.
Bevor es um den eigentlichen Code geht, muss der theoretische Aufbau und Ablauf erläutert werden.
Wie bei vielen anderen Modellen die sich mit maschinellem Lernen befassen sind oft hunterte oder gar hunterttausende Trainingsdurchläufe notwendig, bis das Modell ein zufriedenstellendes Ergebnis liefert. 
Im Fall von GANs handelt es sich dabei um ein sogenanntes <a href="https://en.wikipedia.org/wiki/Nash_equilibrium" style="color: black">Nash-Gleichgewicht</a>, also einen Zustand, bei dem es sich weder für den Diskriminator, noch für den Generator lohnt sich zu verändern.
Welche Herausforderungen damit verbunden sind, wird später [hier](##herausforderungen-beim-gan-training-und-anpassungsmöglichkeiten) erklärt. Zuerst geht es jetzt um die zwei Phasen eines Trainingsdurchlaufs.

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

### Vanishing Gradients
Falls erwünscht, ist es mithilfe von Hyperparametern möglich den ersten und/oder den zweiten Schritt mehrfach hintereinander auszuführen.
Wenn pro Trainingsdurchlauf beispielsweise der erste Schritt häufiger durchlaufen wird als der zweite, so erhält man einen Diskriminator, der deutlich besser echte Daten von unechten Daten des aktuellen Generators unterscheiden kann.
Ein (annähernd) perfekter Diskriminator kann das Vanishing-Gradient-Problem hervorrufen. Bei einem Verlust nahe null passiert es schnell, dass die Gradienten im Bereich des Generators so klein werden, dass sich der Generator quasi nicht verbessern kann.
Tritt dieses Problem zu Beginn des Trainings auf, so bleibt es bei einem vollkommen unbrauchbaren Generator.
Das Auftreten des Problems lässt sich pauschal nicht vorhersagen, jedoch neigt beispielsweise die MinMax-Verlustfunktion dazu. 
Die Wahl einer anderen Verlustfunktion, wie dem Wassersteinverlust, wirken dem Phänomen entgegen [2][2]. 

### Mode Collapse
Mode Collapse beschreibt einen Zustand in dem der Generator Daten ausgibt, die weniger divers sind, als die echten Daten.
Der Generator produziert als beispielsweise nur Bilder von Ananas, obwohl in der Verteilung der echten Daten Ananas, Orangen und Kirschen vorkommen.
Dieses Problem entsteht, wenn es dem Generator deutlich leichter fällt mit einer Klasse von Daten den Diskriminator zu täuschen.
Das Generatortraining führt dann dazu, dass immer mehr solcher Daten erzeugt werden bis letztendlich nur noch Ananasbilder an den Diskriminator übergeben werden.
Nach einer Weile kann es für den Diskriminator günstig sein die Ananasbilder alle abzulehnen. In der Folge ist der Generator gezwungen auf eine neue Frucht zu wechseln.
So werden, wie in <a href="#Abb_5">Abbildung 5</a> zu sehen, nach und nach alle Klassen durchlaufen, ohne das es zu einem universell guten geschweige einem optimalen Ergebnis kommt.
<p align="center">
	<figure>
		<img src="https://github.com/JFJ0831/VIDLMP/blob/5671b345d9edc07654fd0d05b630ede431fff642/10_1.png" title="Abbildung 5" id="Abb_5"/>
		<figcaption>Abbildung 5: Darstellung von Mode Collapse. Target ist die Verteilung der echten Daten, links ist nach unterschiedlich vielen Trainingsdurchläufen zu erkennen, dass der Generator zwischen cerschiedenen Klassen wechselt, es jedoch nicht schafft die gesamte Verteilung zu reproduzieren.</figcaption>
	</figure>
</p>
Auch hier kann die Wassersteinverlustfunktion helfen [2][2].
Daneben gibt es das sogenannte Unrolling [3][3].
Im Gegensatz zum Training des Generators bei einem normalen GAN, werden bei Unrolled GANs nacheinander mehrere neue Gewichte für den Diskriminator berechnet, bevor basierend auf dem letzten Verlust Backpropagation zur Anpassung der Generatorgewichte stattfindet.
Die neuen Generatorgewichte berücksichtigen so zukünftige, noch nicht durchgeführte Anpassungen des Diskriminators. Das Training des Diskriminators bleibt unverändert.
Das Training des Generators sieht demnach im Detail so aus:

	1. Aus zufälligem Rauschen und mit den aktuellen Generatorgewichten neue Daten generieren.
	2. Diskriminator versucht die generierten Daten als solche zu erkennen.
	3. Berechnung des Verlustes.
	4. Erstellen eines neuen Gewichtsvektors für den Diskriminator basierend auf dem alten Vektor und den berechneten Gradienten (Gradientenaufstieg).
	5. Beliebig häufige Wiederholung der Schritte 1-4 mit den aktuellsten Diskriminatorgewichten.
	6. Berechnung des letzten Verlustes.
	7. Backpropagation des Gradienten über alle Wiederholungen.
	8. Anpassung der Generatorgewichte mittels Gradientenabstieg.

<p align="center">
	<figure>
		<img src="" title="Abbildung 6" id="Abb_6"/>
		<figcaption>Abbildung 6: Schematische Darstellung eines dreistufigen unrolled GAN. [3][3]</figcaption>
	</figure>
</p>

### Divergenz/Oszillazion


## Mathematischer Ablauf
Der Generator $G$ soll eine Wahrscheinlichkeitsdichtefunktion $p_g$ erstellen, sodass $G(z)$ mit der gleichen Wahrscheinlichkeit in einem Intervall $[a, b]$ liegt, wie die Zufallsvariable $x$ in einem Intervall $[a, b]$ der Wahrscheinlichkeitsdichtefunktion $p_{data}$.
Der Diskriminator $D$ entscheidet, ob $p_g=p_{data}$, oder nicht. In ersterem Fall ist es dem Generator $G$ gelungen den Diskriminator $D$ zu täuschen.
Der Generator $G$ erhält, wie als Eingabe Werte $z$ aus einer beliebigen Verteilung. Oft handelt es sich dabei, wie bereits geschrieben, um Gaußsches Rauschen, also die Normalverteilung.


Die echten Daten werden mit dem Label $y=1$ und die unechten Daten mit dem Label $y=0$ übergeben.

Auch wird als Label trotz unechter Daten $y=1$ übermittelt.



<div align="center" caption-side="bottom">
	<figure>
		<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_1.png" title="Abbildung 1" width="14%" id="Abb_1"/>
		<figcaption>Abbildung 1</figcaption>
	</figure>
	<figure>
		<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_2.png" title="Abbildung 2" width="14%" id="Abb_2"/>
		<figcaption>Abbildung 2</figcaption>
	</figure>
	<figure>
		<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_3.png" title="Abbildung 3" width="14%" id="Abb_3"/>
		<figcaption>Abbildung 3</figcaption>
	</figure>
	<figure>	
		<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_4.png" title="Abbildung 4" width="14%" id="Abb_4"/>
		<figcaption>Abbildung 4</figcaption>
	<figure>
</div>

<table>
	<tr>
		<th><img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_1.png" title="Abbildung 1" width="100%" id="Abb_1"/></th>
		<th><img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_2.png" title="Abbildung 2" width="100%" id="Abb_1"/></th>
		<th><img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_3.png" title="Abbildung 3" width="100%" id="Abb_1"/></th>
		<th><img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_3.png" title="Abbildung 4" width="100%" id="Abb_1"/></th>
	</tr>
	<tr>
		<th>Abbildung 1</th>
		<th>Abbildung 2</th>
		<th>Abbildung 3</th>
		<th>Abbildung 4</th>
	</tr>
</table>

## Quellen und Referenzen

[1]: https://arxiv.org/pdf/1406.2661.pdf "Generative Adversarial Nets"
[2]: https://arxiv.org/pdf/1701.07875.pdf "Wasserstein GAN"
[3]: https://arxiv.org/pdf/1611.02163.pdf "Unrolled Generative Adversarial Networks"
