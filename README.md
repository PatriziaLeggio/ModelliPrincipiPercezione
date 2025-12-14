# Il Machine Learning nello screening della disgrafia
Progetto esame


La diagnosi di disgrafia avviene ad opera di un'équipe multidisciplinare formata da neuropsichiatra infantile, psicologo e logopedista che devono prima di tutto escludere deficit intellettivi, disturbi neurologici o problemi attentivi importanti.​
Obiettivo dello screening ​
Distinguere tra grafia immatura e disturbo permette di decidere se è necessario un approfondimento diagnostico specialistico.​
​
Nello screening vengono analizzati:​
- Forma delle lettere
  ​
- Dimensione e spaziatura
  ​
- Allineamento sul rigo
  ​
- Pressione sul foglio
  ​
- Postura e impugnatura
  

**Il** **Modello** **Principale** **:** **ResNet18** **(** **Deep** **Learning** **)** ​
Rete Neurale Convoluzionale (CNN) profonda 18 strati. ​
Tecnica usata: Transfer Learning. Invece di partire da zero, il codice scarica i pesi preaddestrati. Questo modello apprende rapidamente e raggiunge un'accuratezza elevata (~90% in validazione).​


**Il** **Modello** **di** **Confronto** **:** **Support** **Vector** **Machine** **(** **Machine** **Learning** **Classico** **)** ​
A differenza della ResNet, l'SVM non guarda l'immagine grezza. Il codice estrae manualmente due caratteristiche matematiche prima di passargliele: Istogramma: La distribuzione dei toni di grigio; Densità: La quantità di pixel neri rispetto al bianco​
Risultati: Come mostra il boxplot l'SVM si ferma al 55% di accuratezza, dimostrandosi inefficace rispetto alla ResNet.​​
