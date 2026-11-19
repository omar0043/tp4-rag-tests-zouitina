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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        logger.setUseParentHandlers(false);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    private static ContentRetriever creerRetriever(
            String fichier,
            EmbeddingModel embeddingModel) {

        Document document = ClassPathDocumentLoader.loadDocument(
                fichier,
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments = DocumentSplitters
                .recursive(1000, 100)
                .split(document);

        List<Embedding> embeddings = embeddingModel
                .embedAll(segments)
                .content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        System.out.println(fichier + " : " + segments.size() + " segments");

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) {
        configureLogger();

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable GEMINI_KEY est absente."
            );
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequests(true)
                .logResponses(true)
                .maxRetries(0)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Chaque PDF possède son propre magasin d'embeddings.
        ContentRetriever retrieverRag = creerRetriever(
                "rag.pdf", embeddingModel
        );

        ContentRetriever retrieverNice = creerRetriever(
                "autre.pdf", embeddingModel
        );

        Map<ContentRetriever, String> descriptions = new LinkedHashMap<>();

        descriptions.put(
                retrieverRag,
                "Cours sur l'intelligence artificielle, les modèles de langage, "
                        + "le fine-tuning, le RAG et les bases vectorielles."
        );

        descriptions.put(
                retrieverNice,
                "Document touristique sur Nice : Promenade des Anglais, "
                        + "Vieux-Nice, Cours Saleya, Colline du Château "
                        + "et proposition de promenade."
        );

        QueryRouter router = new LanguageModelQueryRouter(
                model, descriptions
        );

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
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