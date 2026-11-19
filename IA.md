# Utilisation de l’intelligence artificielle — TP4 RAG

- Étudiant : Omar Zouitina
- Projet : tp4-rag-tests-zouitina
- Date : 19 novembre 2026
- Outil d’assistance : ChatGPT (OpenAI)

## Nature de l’assistance

J’ai utilisé ChatGPT pour m’accompagner dans la réalisation des tests du TP4. L’aide a porté sur l’explication des consignes, la configuration du projet Maven dans IntelliJ, la génération de blocs de code et de classes complètes, la correction d’erreurs et l’interprétation des résultats et des logs. Plusieurs éléments du code ont été fournis par l’IA puis intégrés et exécutés dans mon environnement.

Ce document synthétise cette assistance ; il ne constitue pas une transcription intégrale de la conversation et ne donne pas de pourcentage non mesuré.

## Parties concernées

### RAG naïf — RagNaif.java et Assistant.java

Aide à la rédaction de l’interface Assistant et du programme console : chargement de rag.pdf, découpage en segments, calcul local des embeddings avec AllMiniLmL6V2EmbeddingModel, stockage en mémoire et récupération des segments pertinents. Aide à la configuration de la mémoire conversationnelle, de Gemini et des dépendances Maven.

L’IA a également aidé à corriger l’import du fournisseur Gemini et à comprendre les erreurs de quota. Le modèle utilisé a été changé pour poursuivre les essais avec un modèle disponible dans mon projet.

### Transformation des questions — TestTransformation.java

Aide à l’intégration de CompressingQueryTransformer et de DefaultRetrievalAugmentor, à l’activation des logs et au placement de configureLogger dans la classe Java.

Les essais ont porté sur la question « Qu’est-ce que le fine-tuning ? », suivie de « Dans le support de cours fourni, à quoi c’est comparé ? ». Les logs ont montré la reformulation de la seconde question en une question explicite sur le fine-tuning. Cette reformulation résout l’ambiguïté de la question pour la recherche, mais ne garantit pas à elle seule la pertinence des passages retrouvés ou de la réponse.

### Routage entre deux documents — TestRoutage.java

Aide à la rédaction du code utilisant deux magasins d’embeddings et LanguageModelQueryRouter. ChatGPT a également produit le document complémentaire autre.pdf, consacré à des repères touristiques sur Nice ; rag.pdf est le support de cours de Richard Grin.

Les logs partagés ont permis de vérifier le choix du support de cours pour une question sur le RAG et du document touristique pour une question sur la Colline du Château à Nice.

### Recherche documentaire conditionnelle — Test3.java

Aide à la rédaction d’un routeur personnalisé qui demande au modèle si la question concerne les thèmes du cours et désactive la récupération documentaire pour une réponse « non ».

Les essais et les logs ont montré une décision « non » pour « Bonjour », puis une décision « oui » et une recherche dans rag.pdf pour une question sur l’utilité du RAG.

### Recherche Web et routage — Test4.java

Aide à l’ajout de la dépendance Tavily, à la configuration de la variable d’environnement TAVILY_API_KEY et à la rédaction de la classe combinant le PDF et WebSearchContentRetriever.

Une première version avec DefaultQueryRouter interrogeait les deux sources. Pour une question sur les actualités Artemis de la NASA, les logs contenaient à la fois des résultats Web et des passages du cours sans rapport avec la question.

ChatGPT a proposé de conserver la première version en commentaire et d’utiliser LanguageModelQueryRouter avec une description de chaque source. Après modification, les logs ont montré la sélection de la source 2 (Web) et l’absence de passages de rag.pdf dans le prompt final de cet essai.

La réponse obtenue mélangeait encore des annonces de périodes différentes. L’essai valide le choix de la source pour cette question, pas la fraîcheur ni l’exactitude de toutes les informations générées.

## Vérifications effectuées

J’ai intégré le code dans IntelliJ, lancé les programmes et partagé les sorties et les logs pour analyser leur comportement. J’ai effectué les commits et les push du projet ; l’ajout de ce fichier IA.md a été effectué par l’assistant à ma demande.

Les résultats décrits ci-dessus correspondent aux exécutions partagées dans la conversation. Aucun résultat manquant n’est présenté comme exécuté ou validé. Le modèle Gemini et Tavily sont utilisés par l’application ; ils sont distincts de ChatGPT, utilisé comme aide au développement.

## Limites et précautions

L’analyse des logs ne garantit pas la qualité de toutes les réponses possibles. Le choix des segments, les descriptions des sources et la qualité des résultats Web influencent la réponse. Les clés API ne sont pas reproduites dans ce document et doivent être fournies par des variables d’environnement.
