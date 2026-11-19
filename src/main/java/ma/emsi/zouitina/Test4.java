package ma.emsi.zouitina;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import java.util.LinkedHashMap;
import java.util.Map;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test4 {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        logger.setUseParentHandlers(false);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    private static String lireCle(String nom) {
        String valeur = System.getenv(nom);

        if (valeur == null || valeur.isBlank()) {
            throw new IllegalStateException(
                    "Variable d'environnement absente : " + nom
            );
        }

        return valeur;
    }

    private static ContentRetriever creerRetrieverPdf(
            EmbeddingModel embeddingModel
    ) {
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments = DocumentSplitters
                .recursive(1000, 100)
                .split(document);

        List<Embedding> embeddings = embeddingModel
                .embedAll(segments)
                .content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println(
                "Nombre de segments dans rag.pdf : " + segments.size()
        );

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) {
        configureLogger();

        String geminiKey = lireCle("GEMINI_KEY");
        String tavilyKey = lireCle("TAVILY_API_KEY");

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequests(true)
                .logResponses(true)
                .maxRetries(0)
                .build();

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        ContentRetriever retrieverPdf =
                creerRetrieverPdf(embeddingModel);

        WebSearchEngine moteurWeb = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever retrieverWeb =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(moteurWeb)
                        .maxResults(3)
                        .build();

// Version initiale : interroge systématiquement les deux sources.
// Conservée pour comparer les comportements.
// QueryRouter router = new DefaultQueryRouter(
//         retrieverPdf,
//         retrieverWeb
// );

        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(
                retrieverPdf,
                "Support de cours sur le RAG, le fine-tuning, les embeddings "
                        + "et les modèles de langage. À utiliser pour les questions "
                        + "portant sur ces notions ou sur le support de cours."
        );

        sources.put(
                retrieverWeb,
                "Recherche sur le Web pour les actualités, les événements récents "
                        + "et les sujets absents du support de cours, "
                        + "par exemple les missions spatiales de la NASA."
        );

        QueryRouter router = new LanguageModelQueryRouter(model, sources);

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(router)
                        .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Pose une question, ou écris quitter.");

            while (true) {
                System.out.print("\nQuestion > ");

                if (!scanner.hasNextLine()) {
                    break;
                }

                String question = scanner.nextLine().trim();

                if (question.equalsIgnoreCase("quitter")) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("\nRéponse > " + reponse);
            }
        }
    }
}