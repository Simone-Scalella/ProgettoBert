# Progetto per il corso di Data Science, dove si richiede l'utilizzo di Bert
# Dataset
Per realizzare questo progetto abbiamo utilizzato il dataset Emotion, disponibile al seguente link: 
https://huggingface.co/datasets/emotion.
Emotion è un dataset contenente messaggi Twitter in inglese, etichettati con sei emozioni di base, che sono rabbia, paura, gioia, amore, tristezza e sorpresa. 
Il dataset è composto da 416809 righe e da 2 colonne. La prima colonna rappresenta il messaggio, in ogni riga di questa colonna troviamo un messaggio di Twitter, mentre, nell’altra colonna troviamo le emotions, 
cioè, l’emozione che è stata associata a quel messaggio.
Le emozioni saranno associate a dei numeri durante lo sviluppo di questo progetto.
Le associazioni sono:

 - 0 = Tristezza 
 - 1 = Gioia 
 - 2 = Amore 
 - 3 = Rabbia 
 - 4 = Paura 
 - 5 = Sorpresa
# Sviluppo
Il nostro progetto ha come obiettivo l’utilizzo di Bert per realizzare un modello in grado di eseguire un task di text classification specifico.
Il task di text classification di cui ci siamo occupati è quello del riconoscimento dei sentimenti espressi in un dato testo, quindi, nello specifico, il nostro è un task di sentiment analysis. 
## Fase di ETL
Abbiamo scaricato il dataset dal sito della comunity hugging face e abbiamo deciso di convertirlo in csv per facilitare il resto dello sviluppo e per rendere a noi più comprensibile il dataset in modo tale da essere consci dei dati con cui successivamente abbiamo lavorato. Inizialmente, il dataset era stato salvato come un json, infatti, aveva l’estensione jsonl; è stato utilizzato un breve script per convertire tale file in un csv.
Le emozioni presenti nel dataset sono 6, analizzando il dataset ci siamo resi conto che c’era un piccolo sbilanciamento tra le emozioni. Le classi 4 e 5 risultavano essere più piccole rispetto alle altre. Quindi, si è deciso di addestrare un primo modello con tutte le emozioni tranne la 4 e la 5, e successivamente, si procederà ad addestrare un modello con tutte le emozioni.
Si è proceduto con l’eliminazione delle emozioni meno numerose.

Per migliorare l’addestramento del modello si è deciso di effettuare un’operazione di shuffle sul dataset, in maniera tale da migliorare la distribuzione degli elementi, evitando di avere grandi sequenze di messaggi con la stessa emozione.

Lo step successivo è stato quello di rimuovere le stopwords, ovvero delle parole della frase che non contribuiscono il significato. Quindi per la comprensione, da parte del modello, del testo non sono necessarie, quali ad esempio gli articoli e gli aggettivi possessivi, i caratteri speciali o le emoji.
Questo step va fatto perché essendo scritto il Tweet in linguaggio naturale esistono dei caratteri che ai fini dell’analisi con BERT non sono di alcuna utilità e quindi si decide di rimuoverli.
Si è fatto semplicemente cercando all’interno del testo quelli che venivano definiti come simboli di punteggiatura o come link o come emoji e sostituendoli con uno spazio bianco.

Procederemo ora nel pre-processamento del dataset al fine di renderlo compatibile con il tipo di dato che può essere dato come input a BERT.
BERT, infatti non prende il linguaggio naturale in sé come input, ma prende invece dei tensori (array multidimensionali) formati da token (che derivano dal linguaggio naturale).
Per fare ciò abbiamo in primo creato due liste, una contenente le label (relative al sentimento) e l’altra contenente il testo del Tweet. La lista delle label ci servirà successivamente, in fase di allenamento per far capire al modello ogni Tweet a che sentimento corrisponde.


Prendiamo poi dal BERT tokenizer (che abbiamo importato all’inizio), e con la funzione from_pretrained() il modello 'bert-base-uncased' e do_lower_case=True in quanto come si può vedere nella documentazione questo modello lavora con il testo in lower case.
Il tokenizer sfruttando il modello già allenato è in grado di dividere in token l’input che gli passiamo come frase. 

Inoltre, per rendere i dati compatibili con il tipo di dati accettati in ingresso da BERT ad ogni input vengono assegnati dei token “speciali”, che sono i token [SEP] e [CLS]:

 • [SEP] è un token che se vengono date più frasi in input le separa (ID: 102);
 • [CLS] è un token che va sempre messo all’inizio della frase e specifica che quello che si sta affrontando è un task di classificazione (ID:101).
 
Entrambi i token sono sempre richiesti, anche se la frase in input è una sola (in questo caso [SEP] verrà messo alla fine della frase).
Avendo BERT come massima lunghezza dei token 512, ovvero ogni frase di input dovrà avere al massimo 512 token (parole), dobbiamo quindi andare a vedere se le frasi del nostro dataset rispettano tale limite o meno e in caso non lo rispettassero troncarle.
Sfruttando la funzione encode_plus() facciamo l’encoding di tutte le frasi affinché siano quindi pronte per essere date in input al modello.
Quello che questa funzione fa è:

 • Tokenizzare la frase;
 • Aggiungere i token speciali (ovvero aggiunge i token [SEP] e [CLS]);
 • Mappare i token con il loro ID;
 • Esegue il padding (ovvero allunga aggiungendo 0) o tronca la frase affinché la sua lunghezza sia pari 
a max_length;
 • Creare le attention mask (distinzione tra parole effettive della frase e padding).


Ed infine si procederà a trasformare la lista relativa agli input, la lista relativa alle attention mask e la lista relativa alle label in tensori; a questo punto i nostri dati sono pronti per essere dati in input al modello BERT.
## Realizzazione del modello
Al fine di eseguire il fine tuning si è diviso il dataset in training set e validation set.
Come prima cosa sfruttando la funzione TensorDataset metto insieme i tre 
tensori, relativi agli input, alle attention mask e alle label, ottenuti con le elaborazioni viste prima.
Poi dopo aver definito la dimensione che dovranno avere il training set (80% del dataset) e il validation set (20%) tramite la funzione random_split() vengono creati di due dataset.
Successivamente abbiamo settato i parametri del modello. Per quanto ri-
guarda questi ci sono dei valori “classici” da utilizzare, ma in realtà il modo migliore per selezionarli sarebbe quello di sperimentarli e visionare quale dia il miglior risultato.
Nel nostro progetto, ad esempio, in un primo momento avevamo allenato il modello su 4 epoche, ma vedendo che andava in questo modo in overfitting abbiamo deciso di ridurre il numero delle epoche a 2.
Per caricare i dati sul modello di BERT, dobbiamo utilizzare la classe DataLoader; quindi, creiamo due oggetti DataLoader a cui passeremo rispettivamente il training set e il validation set.
Ed infine procediamo nella creazione del modello, per fare ciò si andrà a riprendere il modello pre-allenato.

Ora dobbiamo procedere nell’allenamento e nella valutazione dei risultati ottenuti.
## Addestramento
Per poter proseguire con l’operazione di fine tuning del modello, sono state realizzate altre operazioni.

 -  Siamo andati a definire il numero di epoche, che, come si può vedere nella precedente immagine, sono state poste a 2. Prima di decidere di prenderne 2 abbiamo anche provato ad allenare il modello con 4 epoche, ma si verificava una condizione di overfitting e perciò abbiamo deciso di limitare il numero di quest’ultime a due. 
 -  Abbiamo creato lo scheduler del learning rate.
 -  Si è definita una funzione per il calcolo dell’accuratezza.


Terminata questa fase di preparazione si è proceduto con il fine tuning del modello utilizzando le varie funzioni messe a disposizione da torch e dalla libreria transformers. 

Al termine dell’addestramento si è proceduto con la fase di test.
## Test
In questa sezione andremo a riportare i risultati ottenuti dall’addestramento dei 4 modelli, divisi per epoche 
e per emozioni. Iniziamo con il modello che classifica quattro emozioni, addestrato per 4 epoche. Riportiamo 
le statistiche legate all’addestramento.

Ulteriori dettagli relativi alle analisi effettuate sono riportati nella seguente [relazione](https://github.com/Simone-Scalella/ProgettoBert/blob/main/Relazione_Bert.pdf)

## Autori
* Caporusso Chiara Amalia: [Github](https://github.com/ChiaraAmalia)
* Galeazzi Margherita: [Github](https://github.com/MargheritaGaleazzi)
* Zhang Yihang: [Github](https://github.com/Accout-Personal)
* Scalella Simone: [Github](https://github.com/Simone-Scalella)
