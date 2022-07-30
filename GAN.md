# Generative Adversarial Networks
*von Julius Joha und Alexander Stahurski im Rahmen des Vertiefungsmoduls "Deep Learning mit Python"*

## Was sind Generative Adversarial Networks?
Generative Adversarial Networks, kurz GAN, gehören zu den generativen Modellen. Diese Modelle sind in der Lage neue Daten, also beispielsweise Bilder von fiktiven Gesichtern oder Häusern, zu erstellen.
GANs bestehen aus zwei neuronalen Netzen welche gegeneinander antreten.
Eins der beiden Netze, Generator genannt, versucht dabei echte Daten so gut zu imitieren, dass das andere Teilnetz, der Discriminator, nicht in der Lage ist, die neu generierten Daten von echten Daten zu unterscheiden.

## Wozu werden GANs verwendet?
Im ursprünglichen Forschungspaper von Ian Goodfellow et al. [^1] wurde ein GAN erstellt, welches Bilder von Zahlen generieren kann, die so aussehen, wie die aus dem MNIST-Datensatz.
Mittlerweile wurden GANs in einer Vielzahl von Anwendungsgebieten erfolgreich eingesetzt. Dazu sei jedoch gesagt, dass die verwendeten Modelle deutlich komplexer geworden sind, als das ursprüngliche und hier vorwiegend behandelte.
Die prominenteste Verwendung für GANs ist wohl im Bereich der Bildgenerierung und insbesondere der Generierung von Gesichtern nicht existierender Personen angesiedelt.
In diesem Bereich nehmen die sogenannten StyleGANs eine Vorreiterrolle ein. Auf die wesentlichen Merkmale dieser GANs gehen wir [hier](user-content-stylegan) ein.

## Wie funktionieren GANs?
Im diesem Blog sollen GANs vorgestellt werden welche zum Beispiel in der Lage ist echt aussehende Bilder von Früchten zu generieren.
Wie bei vielen anderen Modellen die sich mit maschinellem Lernen befassen sind oft hunterte oder gar hunterttausende Trainingsdurchläufe notwendig, bis das Modell ein zufriedenstellendes Ergebnis liefert. 
Im Fall von GANs handelt es sich dabei um ein sogenanntes <a href="https://en.wikipedia.org/wiki/Nash_equilibrium" style="color: black">Nash-Gleichgewicht</a>, also einen Zustand, bei dem es sich weder für den Diskriminator, noch für den Generator lohnt sich zu verändern.
Welche Herausforderungen damit verbunden sind, wird später [hier](user-content-herausforderungen-beim-gan-training-und-anpassungsmöglichkeiten) erklärt. Zuerst geht es jetzt um die zwei Phasen eines Trainingsdurchlaufs.

### Erste Phase
Der vordere Teil des GANs, der Generator, wird mit einem zufälligen Input gefüttert. In vielen Fällen handelt es sich dabei um gaußsches Rauschen, aber auch andere Verteilungen sind möglich.
Den Input bearbeitet der Generator nun bestenfalls so, dass echt aussehende Bilder von Früchten entstehen. In den ersten Durchläufen wird dies nicht gelingen, der Output wird mehr oder weniger zufällig sein.
Dies liegt dran, dass die Gewichte des neuronalen Netzwerkes zufällig initialisiert wurden und auch noch nicht ausreichend in eine Richtung angepasst werden konnten, die es ermöglicht echt aussehende Daten zu generieren.
Bevor die Gewichte des Generators überhaupt angepasst werden, wird zunächst der Discriminator trainiert.
Dieser wird dazu zeitgleich sowohl mit gerade neu generierten Daten, aber auch mit echten Daten gefüttert. Die Daten kommen dabei jeweils in identischer Anzahl vor.
Der Discriminator versucht nun die Daten richtig zuzuordnen. Wie erfolgreich diese geschieht, lässt sich mithilfe der MinMax-Verlustfunktion errechnen.
Nun gilt es festzustellen, welche Kantengewichte wie angepasst werden müssen, um den Verlust zu reduzieren.
Dazu werden unter Zuhilfenahme von Backpropagation entsprechende Gradienten berechnet und dann die Gewichte der Kanten in die entgegengesetzte Richtung der Gradienten angepasst (-> stochastischer Gradientenabstieg). 
Zu Beachten ist, dass während des Trainings des Discriminators die Gewichte des Generators nicht verändert werden. Dieser Teil des GANs wird in dieser Phase sogar komplett ausgeklammert.

### Zweite Phase
Sind die Gewichte des Discriminators aktualisiert worden, so widmet sich der nächste Schritt dem Training des Generators.
Auch hier wird der Generator zuerst mit Zufallseingaben beschickt, welche dann zu möglichst realistischen Daten modifiziert werden.
Die Unterscheidung zur ersten Phase liegt unter anderem darin, dass der Discriminator jetzt nicht sowohl mit echten als auch unechten Daten gefüttert wird, sondern nur mit unechten.
Der Discriminator versucht nun möglichst alle Daten als unecht zu erkennen. Auch hier wird ein Verlust berechnet, wobei die Funktion eine andere sein kann als beim Training des Discriminators. 
Der Generator wird bestraft, wenn der Discriminator ein generiertes Bild als solches erkennt.
Um die Gewichte des Generators anzupassen, muss die Backpropagation des Verlustes folglich über das gesamte GAN erfolgen.
Negative Auswirkungen auf die Leistung des Discriminators lassen sich unterbinden, indem dessen Gewichte zuvor festgesetzt werden und für diesen Schritt unveränderlich sind.
Nach Aktualisierung der Generatorgewichte kann ein neuer Durchlauf begonnen werden.

Falls erwünscht, ist es mithilfe von Hyperparametern möglich die erste und/oder die zweite Phase mehrfach hintereinander auszuführen.
Wenn pro Trainingsdurchlauf beispielsweise die erste Phase häufiger durchlaufen wird als die zweite, so erhält man einen Diskriminator, der deutlich besser echte Daten von unechten Daten des aktuellen Generators unterscheiden kann.

## Herausforderungen beim GAN-Training und Anpassungsmöglichkeiten

### Vanishing Gradients
Ein (annähernd) perfekter Diskriminator kann das Vanishing-Gradient-Problem hervorrufen. Bei einem Verlust nahe null passiert es schnell, dass die Gradienten im Bereich des Generators so klein werden, dass sich der Generator quasi nicht verbessern kann.
Tritt dieses Problem zu Beginn des Trainings auf, so bleibt es bei einem vollkommen unbrauchbaren Generator.
Das Auftreten des Problems lässt sich pauschal nicht vorhersagen, jedoch neigt beispielsweise die MinMax-Verlustfunktion dazu. 
Die Wahl einer anderen Verlustfunktion, wie dem <a href="https://en.wikipedia.org/wiki/Wasserstein_metric">Wassersteinverlust</a>, wirken dem Phänomen entgegen [^2]. 

### Instabilität/Oszillazion
Das Training von GANs gestaltet sich schwierig, da die beiden Teilnetze gegeneinander arbeiten. Die Eingaben können unzählige Dimensionen haben und die Kostenfunktion ist meist nicht konvex. [^5]
Das Erreichen des Nash-Gleichgewichts ist deshalb keinesfalls garantiert. Das Training kann auf zahlreiche Arten beeinflusst werden.
Die Wahl der Hyperparameter kann den Unterschied zwischen einem konvergierenden GAN machen und einem GAN welches mit dem Training gar nicht erst loslegt.


### Mode Collapse
Mode Collapse beschreibt einen Zustand in dem der Generator Daten ausgibt, die weniger divers sind, als die echten Daten.
Der Generator produziert als beispielsweise nur Bilder von Ananas, obwohl in der Verteilung der echten Daten Ananas, Orangen und Kirschen vorkommen.
Dieses Problem entsteht, wenn es dem Generator deutlich leichter fällt mit einer Klasse von Daten den Diskriminator zu täuschen.
Das Generatortraining führt dann dazu, dass immer mehr solcher Daten erzeugt werden bis letztendlich nur noch Ananasbilder an den Diskriminator übergeben werden.
Nach einer Weile kann es für den Diskriminator günstig sein die Ananasbilder alle abzulehnen. In der Folge ist der Generator gezwungen auf eine neue Frucht zu wechseln.
So werden, wie in Abbildung 5 zu sehen, nach und nach alle Klassen durchlaufen, ohne das es zu einem universell guten geschweige einem optimalen Ergebnis kommt.

![Abbildung 5](https://github.com/JFJ0831/VIDLMP/blob/5671b345d9edc07654fd0d05b630ede431fff642/10_1.png)
*Abbildung 5: Darstellung von Mode Collapse. Target ist die Verteilung der echten Daten, links ist nach unterschiedlich vielen Trainingsdurchläufen zu erkennen, dass der Generator zwischen cerschiedenen Klassen wechselt, es jedoch nicht schafft die gesamte Verteilung zu reproduzieren. [^3]*

Auch hier kann die Wassersteinverlustfunktion helfen [^2].
Daneben gibt es das sogenannte Unrolling [^3].
Im Gegensatz zum Training des Generators bei einem normalen GAN, werden bei Unrolled GANs nacheinander mehrere neue Gewichte für den Diskriminator berechnet, bevor basierend auf dem letzten Verlust Backpropagation zur Anpassung der Generatorgewichte stattfindet.
Die neuen Generatorgewichte berücksichtigen so zukünftige, noch nicht durchgeführte Anpassungen des Diskriminators. Das Training des Diskriminators bleibt unverändert.
Das Training des Generators sieht im Detail so aus:

	1. Aus zufälligem Rauschen und mit den aktuellen Generatorgewichten neue Daten generieren.
	2. Diskriminator versucht die generierten Daten als solche zu erkennen.
	3. Berechnung des Verlustes.
	4. Erstellen eines neuen Gewichtsvektors für den Diskriminator basierend auf dem alten Vektor und den berechneten Gradienten (Gradientenaufstieg).
	5. Beliebig häufige Wiederholung der Schritte 1-4 mit den aktuellsten Diskriminatorgewichten.
	6. Berechnung des letzten Verlustes.
	7. Backpropagation des Gradienten über alle Wiederholungen.
	8. Anpassung der Generatorgewichte mittels Gradientenabstieg.

![Abbildung 6](https://github.com/JFJ0831/VIDLMP/blob/aac187f9b75607901f55e5c9ee4f13fbd43b2daf/11.png)
*Abbildung 6: Schematische Darstellung eines dreistufigen unrolled GAN. [^3]*

![Abbildung 7](https://github.com/JFJ0831/VIDLMP/blob/e0cbfa4867d85a942a6a61519517a7567a251119/10_2.png)
*Abbildung 7: Heatmap der von einem zehnstufigen unrolled GAN generierten Verteilungen. [^3]*

Außerdem lässt sich Mode Collapse umgehen, indem man nicht nur generierte Daten des aktuellen Generators an den Diskriminator übergibt, sondern auch Daten die einem vorherigen Generator entstammen.
Der Diskriminator passt sich so nicht übermäßig an den aktuellen Generator an, sondern auch an vorherige. Ausgaben aktuellerer Generatoren sind dabei meist relevanter als älterer. [^11]

Ein letztes hier vorgestelltes Mittel zu Verhinderung von Mode Collapse ist die mini-batch discrimination. Hierbei wird errechnet, wie ähnlich die Instanzen innerhalb eines Batch sind.
Da beim Mode Collapse wenig diverse Daten erzeugt werden, kann der Diskriminator durch diese zusätzliche Information ein solches Batch komplett ablehnen. [^5]



## Mathematischer Ablauf
Der Generator $G$ soll $G(z)$ so ausgeben, dass diese mit der gleichen Wahrscheinlichkeit in einem Intervall $[a, b]$ der dadurch entstehenden Wahrscheinlichkeitsdichtefunktion $p_g$ liegen, 
wie echte Daten $x$ aus einem Intervall $[a, b]$ der Wahrscheinlichkeitsdichtefunktion $p_{data}$ kommen.
Der Diskriminator $D$ entscheidet, ob $p_g(x)=p_{data}(x)$, oder nicht. In ersterem Fall ist es dem Generator $G$ gelungen den Diskriminator $D$ zu täuschen.
Der Generator $G$ erhält, als Eingabe Werte $z$ aus einer beliebigen Verteilung $p_z$. Oft handelt es sich dabei, wie bereits erwähnt, um Gaußsches Rauschen, also die Normalverteilung.
Aus dieser Eingabe $z$ wird eine Ausgabe $G(z)$ aus der dem Generator zugrunde liegenden Wahrscheinlichkeitsdichte $p_g$ generiert. $p_g$ hat die selbe Definitionsmenge wie $p_{data}$.
Es handelt sich beim Diskriminator $D$ um ein binäres Klassifikationsnetzwerk. Das neuronale Netzwerk wird in diesem Fall auf die MinMax-Verlustfunktion optimiert:

$$ \min_G \max_D⁡ V(D,G) = \mathbb{E}\_{x \sim p_{data}(x)} [ \log ⁡D(x)] + \mathbb{E}_{z \sim p_z (z)} [ \log⁡ (1-D(G(z))) ] $$

Diese Funktion ähnelt stark der binären Kreuzentropie mit $n=1$ Instanzen:

$$ \mathbb{L} = - \sum_{i=1}^n y_i \log (\hat{y_i}) + (1-y_i) \log (1-\hat{y_i}) $$

Um beispeilsweise in Keras `binary_crossentropy` als Verlustfunktion zu nutzen, müssen beim Training des Diskriminators die echten Daten $x$ mit dem Label $y=1$ und die unechten Daten $G(z)$ mit dem Label $y=0$ an $D$ übergeben werden.
Folglich kann ein so programmiertes GAN entweder eine Instanz von echten Daten oder eine Instanz von unechten Daten entgegen nehmen.
Der Diskriminator des originalen Algorithmus [^1] kann aufgrund der geringfügig anderen Verlustfunktionen gleichzeitig echte und unechte Daten verarbeiten. 

Bei der normalen binären Kreuzentropie gilt für $y=0, \hat{y}=D(G(z))$: $\mathbb{L} = - \log (1-D(G(z)))$ und für $y=1, \hat{y}=D(x)$ gilt $\mathbb{L} = - \log (D(x))$.

Addiert man die beiden Fälle, ergibt sich daraus $\mathbb{L} = - \log (D(x)) - \log (1-D(G(z)))$.

Diese Funktion soll minimiert werden. Eine Funktion kann stattdessen mit $-1$ multipliziert und dann maximiert werden um zum gleichen Ergebnis zu gelangen.
Außerdem soll betrachtet werden, wie der Verlust über einen gesamten Datensatz ist. Dazu wird der Erwartungswert berechnet. Die dann daraus resultierende Funktion ist identisch mit $V(D,G)$.

$$ \mathbb{E}(\mathbb{L}) = \int p_{data}(x) \log ⁡D(x)dx + \int p_z (z) \log⁡ (1-D(G(z))dz = V(D,G) $$

$D$ ist dazu angehalten die Funktion $V(D,G)$ zu maximieren. 
Dies ist der Fall, wenn $D$ echte Daten $x$ als solche erkennt, also $D(x)$ möglichst gegen $1$ geht. $\log D(x)$ nähert sich dann von unten gegen $0$.
Außerdem sollten generierte Daten $G(z)$ als solche erkannt werden. $D$ muss diesen Daten dementsprechend eine geringe Wahrscheinlichkeit zusprechen zur Klasse der echten Daten zu gehören.
$D(G(z))$ sollte also gegen $0$ gehen, womit $\log⁡ (1-D(G(z)))$ sich dann ebenfalls von unten der $0$ nähert.
Ein Diskriminator $D$ der einen Wert von $0$ erreicht wäre somit optimal, während ein schlechter Diskriminator hohe negative Werte erzielen würde.
Die Gewichte $\theta_d$ werden nun entsprechend angepasst. Dazu wird der Gradient $\nabla_{\theta_d}$ berechnet. Je nachdem ob $V(D,G)$ oder `binary_crossentropy` verwendet wird, muss ein Gradientenauf- oder abstieg durchgeführt werden.


Der Generator $G$ versucht den Verlust $V(D,G)$ zu minimieren.
Dazu muss er möglichst $G(z)$ generieren, die $D$ nicht als solche erkennt.
$D(G(z))$ muss also möglichst gegen $1$ gehen, damit $\log (1-D(G(z)))$ möglichst hohe negative Werte annehmen kann.
Beim Training des Generators werden nur unechte Daten übergeben.

Bei der Nutzung von `binary_crossentropy` für den Generator in Keras geschieht dies jedoch mit dem Label $y=1$ statt wie beim Diskriminator mit dem Label $y=0$.
Der Grund dafür liegt darin, dass jetzt die vom Generator generierten Daten vom Diskriminator so behandelt werden, wie dieser auch echte behandelt und insbesondere weil für $y=1$, wie zuvor erläutert, der Verlust anders berechnet wird, als für $y=0$.
Die Funktion lautet dann nämlich $\mathbb{L} = - \log (\hat{y})$. Wobei hier $\hat{y}=D(G(z))$ ist.
Je größer $D$ die Wahrscheinlichkeit einschätzt, dass die generierten Daten echt sind, desto niedriger der Wert dieser Funktion und desto besser $G$.
Damit wird sowohl bei der Nutzung von $V(D,G)$ als auch bei der Nutzung der binären Kreuzentropie das selbe Optimum erreicht.

In beiden Fällen werden die Generatorgewichte $\theta_g$ mithilfe des Gradienten $\nabla_{\theta_g}$ und dem Gradientenabstieg angepasst.

Das Nash-Gleichgewicht ist erreicht, wenn die Verteilung der echten Daten identisch ist mit der Verteilung der generierten Daten, also $p_{data}(x)=p_{g}(x)$ [^1].
Der Diskriminator klassifiziert dann zu 50% richtig. 


In den folgenden Abbildungen ist schematisch dargestellt, wie sich das GAN und seine Netzwerkteile entwickeln um schlussendlich das Nash-Gleichgewicht zu erreichen.
Ganz unten in den Abbildungen ist die Verteilung $p_z$ zu erkennen, aus der $z$ gezogen werden, die dann durch den Generator (schwarze Pfeile) zu $x$ werden.
Die grüne Linie gibt die Wahrscheinlichkeitsdichtefunktion $p_g$ der generierten Daten an. Die schwarz gepunktete Linie bildet $p_{data}$ ab.
Die blau gestrichelte Linie stellt dar, mit welcher Wahrscheinlichkeit ein $x$ als echt erkannt wird.

|<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_1.png" title="Abbildung 1" width="180" id="Abb_1"/>|<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_2.png" title="Abbildung 2" width="180" id="Abb_2"/>|<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_3.png" title="Abbildung 3" width="180" id="Abb_3"/>|<img src="https://github.com/JFJ0831/VIDLMP/blob/8775769721fbca1ca9c5ed038a3db14863064016/08_3.png" title="Abbildung 4" width="180" id="Abb_4"/>|
|---|---|---|---|
|*Abbildung 1:<br />Der Diskriminator ist noch untrainiert. Die Wahrscheinlichkeit einer korrekten Klassifikation springt relativ stark.[^1]*|*Abbildung 2:<br />Der Diskriminator wurde trainiert. Links werden Daten eher als echt eingestuft als rechts, wo zu erkennen ist, dass die generierten Daten anders verteilt sind, als die echten Daten.[^1]*|*Abbildung 3:<br />Der Generator wurde traniert. Die Verteilung der unechten Daten nähert sich der Verteilung der echten Daten an.[^1]*|*Abbildung 4:<br />Nach mehreren Durchläufen konvergiert das Netzwerk. Die Verteilungen sind nicht mehr zu unterscheiden.[^1]*|



## StyleGAN

Das GAN ist in soweit beschränkt, dass wir über die Merkmale, sagen wir hier innerhalb eines generierten Bildes, keinen direkten Einfluss haben. Es ist uns nur möglich über die Trainingsdaten die generierten Bilder in eine gewisse Richtung zu lenken. Aber vor allem je kleiner das gewünschte Merkmal ist, dass man beim generieren verändern oder hinzufügen möchte, desto schwieriger wird es auch darauf über die Trainingsdaten Einfluss zu nehmen.
Darüberhinaus ist es auch eher eine unzuverlässige Methode, da wir nicht sicherstellen können, dass die Merkmale über die Trainingsdaten auch tatsächlich wie gewünscht so generiert werden. 
Es ist eher unflexibel, Trainingsdauer ist dementsprechend theoretisch unendlich, also es ist auch nicht sicher, ob das gewünschte Ergebnis überhaupt erreicht wird.
Das heißt wir brauchen eine andere Option genau das zu bewerkstelligen, noch präziser und (hoffentlich) schneller.

Kommen wir nun zur Architektur.

![StyleGAN Architekturüberblick](https://github.com/JFJ0831/VIDLMP/blob/37189f8429e432224a7ee94687ff009a734be4a3/12.jpg)

*Abbildung 7: Überblick über die StyleGAN Architektur [^12]*

### Mapping Network
Bevor das eigentliche Bild generiert wird, erstellen wir vorher einen Vektor, der es uns ermöglicht Stellen innerhalb des Bildes, nehmen wir hier Gesichter als Anhaltspunkt, präziser zu manipulieren.

Zunächst wird der latent code z, also beispielsweise Merkmale innerhalb eines Gesichtes, gemapt durch ein acht-layer MLP, woraus ein Vektor w entsteht. Dieser Vektor wird an verschiedenen Stellen innerhalb des Generators eingefügt, aber diese sind nicht an jeder Stelle gleich. Der Vektor w wird durch verschiedene Stellen (repräsentiert durch die Boxen A) durch beispielsweise Dense Layer ohne Aktivierungsfunktion geschickt, woraus dann mehrere Vektoren entstehen, auch Style Vektoren gennant [^11].
Diese sind dann in der Lage, zusammen mit der Adaptive Instance Normalization (AdaIN), verschiedene Merkmale innerhalb eines Bildes präzise zu ändern, auf verschiedensten Ebenen. Z.B. schafft es die Möglichkeit, die Gesichtsform an sich zu ändern, als auch kleinere Falten im Gesicht einzufügen, aber dabei andere Elemente im Gesicht in der Form zu erhalten (z.B. Haarfarbe, Augenfarbe, Größe der Nase etc.) [^11].

### Rauschen
Das Rauschen im StyleGAN hat eine etwas andere Rolle als bei einem herkömmlichen GAN. Das Rauschen erfüllt hier verschiedene Zwecke:
* Kein vom Generator oder vom latent code künstlich erzeugter Zufall (z.B. Krümmung vom Haar)
    * Heißt wir brauchen keine Methode diesen erzeugten Zufall zu speichern, was sonst einen nicht unerheblichen Teil der Leistung des Generators beanspruchen würde
* Einfügen an verschiedenen Stellen innerhalb des Generators, auch in höheren Ebenen
    * Das Rauschen müsste ansonsten durch das ganze Netzwerk, was die Trainingsdauer erheblich anheben könnte.
* Das Rauschen ist nicht an jeder eingefügten Stelle gleich
    * Ohne unterschiedliches Rauschen enstehen möglicherweise visuelle Artefakte, da das gleiche Rauschen über die verschiedenen Ebenen verwendet wird.
    
Das Rauschen wird als Input und Output jedes Convolutional Layers eingefügt, aber vor der Aktivierungsfunktion und wird gefolgt von einer Adaptive Instance Normalization.

Bei dem Generator nehmen wir am Anfang also keinen vom gausschen' Rauschen generiertes Bild, sondern nehmen eine gelernte Konstante. Diese Konstante ist erst wirklich "konstant" nach dem Training und wird vorher durch Backpropagation [^11] kontinuierlich angepasst. Das heißt, falls man einen eigenen StyleGAN trainieren sollte, erhält man also möglicherweise eine andere als hier angegebene Konstante.
Das Rauschen besteht aus einer Feature Map, welche auf die Feature Maps in der jeweiligen Ebene übertragen wird (siehe Architektur). Vor der Übertragung werden diese Feature Maps skaliert über gelernte per-feature Skalierungsfaktoren [^12][^11] (dargestellt als Box B).

### Adaptive Instance Normalization

Bevor wir näher auf die Struktur vom Generator eingehen, ist es wichtig zu wissen was *Adaptive Instance Normalization* $AdaIN$ ist.
$AdaIN$ ist eine weitere Version von *Instance Normalization* und diese wiederrum ist eine weitere Version von der *Batch Normalization* $BN$.

$$ BN(x) = 𝛾\frac{x-𝜇(x)}{𝜎(x)}+𝛽 $$

Mit 𝛾 als Skalierungsfaktor, 𝛽 als Offset, 𝜇(x) als Durchschnitt der Feature Maps und 𝜎(x) als deren Standardabweichung. 
Daraus folgt dann die **Instance Normalization** $IN$.

$$ IN(x_i) = 𝛾\frac{x_i-𝜇(x_i)}{𝜎(x_i)}+𝛽 $$

Diese Erweiterung der Batch Normalization ermöglicht es, aus einem Batch aus Feature Maps, jede Feature Map einzelnd zu normalisieren, anstelle eines einheitlichen Parameters, den man für einen ganzen Batch nutzen würde. Für den StyleGAN reicht das aber noch nicht ganz aus, um präzise Einfluss auf die Merkmale nehmen zu können.
Mit **Adaptive Instance Normalization** ist es aber möglich, da mit AdaIn die Skalierungsfaktoren vom Style Vektor ausgehen. Dieser Vektor enthält je einen Skalierungsfaktor und Offset [^11] pro enthaltener Feature Map. Das ermöglicht Einfluss auf Details innerhalb eines Gesichtes, ohne andere Details damit zu beeinflussen.

$$ AdaIN(x_i,y) = 𝜎(y)\frac{x_i-𝜇(x_i)}{𝜎(x_i)}+𝜇(y) $$

Mit y als korrespondierende skalierte Style-Komponente [^12][^13] von den Style Vektoren. 

### Synthesis Network / Generator

Der Generator ist relativ simpel aufgebaut. Wir starten mit einer niedrigen Größe (4x4) und Upsampeln die Größe zu/vor jeder neuen Ebene (8x8, 16x16 etc.). Wie zuvor genannt, beginnt der Generator nicht mit einem zufälligem Rauschen, sondern einer gelernten Konstante. Das zuvor beschriebene Rauschen wird zum Output eines Convolutional Layers hinzugefügt, damit jedes generierte Bild eine zufällige, aber natürliche Komponente bekommt. Dieser Output wird dann durch Adaptive Instance Normalization normalisiert, verändert durch den Style und als Input für einen weiteren Convolutional Layer verwendet, wo der Ablauf sich dementsprechend wiederholt. 

Pro Ebene gibt es also zwei Convolutional Layer, zwei Adaptive Instance Normalization Layer, zwei Rauschvektoren und ein Upsampling Layer zu jeder neuen Ebene.

Auf den tieferen Ebenen (4x4, 8x8) verändern sich grundlegende Merkmale wie z.B. Gesichtsform, Haltung etc. und je höher die Ebene, desto mehr werden eher die detailierteren Aspekte verändert, wie z.B. Haarfarbe oder Hautporen auf den letzten Ebenen [^12].


### Style Mixing

Darüber hinaus verwendet der StyleGAN Style Mixing. Es werden mehrere latent codes gemapt, woraus dann mehr Style Vektoren entstehen. Diese werden dann auf zufälligen Ebenen [^12][^11] verwendet, damit der Generator nicht denkt, dass die Styles bei benachbarten Ebenen eine Korrelation haben. Das hat verschiedene Vorteile:
* Zum einen fördert es Lokalisierung im Generator, heißt dass jeder Style Vektor noch weniger Merkmale gleichzeitig manipuliert [^12][^11]
* Zum anderen schneidet mit Style Mixing der StyleGAN besser ab in einem GAN Bewertungstest, als ohne Style Mixing [^12]


[^1]: https://arxiv.org/pdf/1406.2661.pdf "Generative Adversarial Nets"
[^2]: https://arxiv.org/pdf/1701.07875.pdf "Wasserstein GAN"
[^3]: https://arxiv.org/pdf/1611.02163.pdf "Unrolled Generative Adversarial Networks"
[^5]: https://arxiv.org/pdf/1606.03498.pdf "Improved Techniques for Training GANs"
[^11]: Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow : Concepts, Tools, and Techniques to Build Intelligent Systems, O'Reilly Media, Incorporated, 2019.
[^12]: Tero Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks", arXiv pre‐print arXiv:1812.04948 (2018). https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf
[^13]: Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." Proceedings of the IEEE international conference on computer vision. 2017.
