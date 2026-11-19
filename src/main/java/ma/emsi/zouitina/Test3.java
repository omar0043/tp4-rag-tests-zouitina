package ma.emsi.zouitina;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3 {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        logger.setUseParentHandlers(false);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("GEMINI_KEY est absente.");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequests(true)
                .logResponses(true)
                .maxRetries(0)
                .build();

        // Ingestion du seul PDF consacré au RAG.
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments = DocumentSplitters
                .recursive(1000, 100)
                .split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddingModel.embedAll(segments).content(), segments);

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        PromptTemplate template = PromptTemplate.from("""
                Est-ce que la requête suivante porte sur l'intelligence
                artificielle, les modèles de langage, le fine-tuning,
                le RAG ou les bases vectorielles ?

                Une simple salutation ne porte pas sur ces sujets.

                Requête :
                {{question}}

                Réponds seulement par oui, non ou peut-être.
                """);

        // Routeur personnalisé : décider d'utiliser le RAG ou de l'éviter.
        QueryRouter router = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                String prompt = template
                        .apply(Map.of("question", query.text()))
                        .text();

                String decision = model.chat(prompt)
                        .trim()
                        .toLowerCase(Locale.ROOT);

                System.out.println("Décision du routeur : " + decision);

                if (decision.equals("non")) {
                    System.out.println("Recherche documentaire désactivée.");
                    return List.of();
                }

                System.out.println("Recherche dans rag.pdf activée.");
                return List.of(retriever);
            }
        };

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